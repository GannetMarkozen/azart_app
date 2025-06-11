pub use futures;

use std::cell::{Cell, OnceCell, RefCell, UnsafeCell};
use std::collections::VecDeque;
use std::sync::{Arc, Condvar, Mutex, MutexGuard, OnceLock, RwLock};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::{env, mem, thread};
use mem::ManuallyDrop;
use std::array::from_fn;
use std::fmt::{Debug, Formatter};
use std::marker::{PhantomData, PhantomPinned};
use std::pin::{pin, Pin};
use std::rc::Rc;
use std::task::{Context, Poll, RawWaker, RawWakerVTable, Waker};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};
use crossbeam::deque::{Worker, Stealer, Steal};
use crossbeam::queue::ArrayQueue;
use crossbeam::sync::{Parker, Unparker};
use crossbeam::utils::CachePadded;
use futures::{pin_mut, FutureExt};
use futures::executor::block_on;
use futures::future::Shared;
use rand::prelude::IndexedRandom;
use rand::rng;
use scopeguard::defer;

static EXECUTOR: OnceLock<Executor> = OnceLock::new();
thread_local! {
	static LOCAL: OnceCell<ThreadLocalState> = const { OnceCell::new() };
	static NESTED_CALLS: Cell<usize> = const { Cell::new(0) };
}

#[derive(Debug)]
#[must_use = "Tasks are cancelled on Drop! Use await or detach() or cancel() explicitly!"]
pub struct Task<T>(async_task::Task<T>);

impl<T> Task<T> {
	#[inline(always)]
	pub fn spawn<F>(f: F) -> Self
		where
			T: Send + 'static,
			F: Future<Output = T> + Send + 'static,
	{
		Builder::new().spawn(f)
	}
	
	#[inline(always)]
	pub fn spawn_local<F>(f: F) -> Self
		where
			T: 'static,
			F: Future<Output = T> + 'static,
	{
		Builder::new().spawn_local(f)
	}
	
	#[inline(always)]
	pub async fn cancel(self) -> Option<T> {
		self.0.cancel().await
	}
	
	#[inline(always)]
	pub fn detach(self) {
		self.0.detach();
	}
	
	#[inline(always)]
	pub fn is_finished(&self) -> bool {
		self.0.is_finished()
	}
}

impl<T> Future for Task<T> {
	type Output = T;
	fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
		pin!(&mut self.0).poll(cx)
	}
}

#[derive(Debug)]
#[must_use = "Not cancellable! Will run to completion upon drop!"]
pub struct ScopedTask<T>(ManuallyDrop<Task<T>>);

impl<T> Future for ScopedTask<T> {
	type Output = T;
	fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
		pin!(&mut *self.0).poll(cx)
	}
}

impl<T> Drop for ScopedTask<T> {
	fn drop(&mut self) {
		Executor::get().block_on(unsafe {
			ManuallyDrop::take(&mut self.0)
				.0
				.fallible()// Must be made fallible because the task could have already been polled to completion.
		});
	}
}

#[must_use = "Must use to create Task!"]
#[derive(Default, Clone, Debug)]
pub struct Builder {
	pub priority: Priority,
	pub target: Target,
}

impl Builder {
	pub fn spawn<F>(self, f: F) -> Task<F::Output>
		where
			F: Future + Send + 'static,
			F::Output: Send + 'static,
	{
		let (runnable, task) = match self.target {
			Target::Any => async_task::spawn(
				f,
				match self.priority {
					Priority::High => |r| Executor::get().push(r, Priority::High, Target::Any),
					Priority::Normal => |r| Executor::get().push(r, Priority::Normal, Target::Any),
					Priority::Low => |r| Executor::get().push(r, Priority::Low, Target::Any),
				},
			),
			Target::Thread(thread) => match self.priority {
				Priority::High => async_task::spawn(f, move |r| Executor::get().push(r, Priority::High, Target::Thread(thread))),
				Priority::Normal => async_task::spawn(f, move |r| Executor::get().push(r, Priority::Normal, Target::Thread(thread))),
				Priority::Low => async_task::spawn(f, move |r| Executor::get().push(r, Priority::Low, Target::Thread(thread))),
			},
		};

		runnable.schedule();
		
		Task(task)
	}
	
	pub fn spawn_local<F>(self, f: F) -> Task<F::Output>
		where
			F: Future + 'static,
			F::Output: 'static,
	{
		let (runnable, task) = async_task::spawn_local(
			f,
			match self.priority {
				Priority::High => |r| Executor::get().push(r, Priority::High, Target::Thread(ThreadId::current())),
				Priority::Normal => |r| Executor::get().push(r, Priority::Normal, Target::Thread(ThreadId::current())),
				Priority::Low => |r| Executor::get().push(r, Priority::Low, Target::Thread(ThreadId::current())),
			},
		);

		runnable.schedule();

		Task(task)
	}
	
	pub fn spawn_scoped<'a, F>(self, scope: &'a Scope, f: F) -> ScopedTask<F::Output>
		where
			F: Future + Send + 'a,
			F::Output: Send + 'a,
	{
		scope.add_task();
		
		let task = unsafe {
			self.spawn_unchecked(async move {
				let scope = &*(scope as *const Scope);

				let result = f.await;
				scope.complete_task();
				result
			})
		};
		
		ScopedTask(ManuallyDrop::new(task))
	}
	
	pub fn spawn_local_scoped<'a, F>(self, scope: &'a Scope, f: F) -> ScopedTask<F::Output>
		where
			F: Future + 'a,
	{
		scope.add_task();

		let task = unsafe {
			self.spawn_local_unchecked(async move {
				let result = f.await;
				scope.complete_task();
				result
			})
		};

		ScopedTask(ManuallyDrop::new(task))
	}
		
	/// # Safety
	/// 
	/// Lifetimes of captures can not be statically verified!
	pub unsafe fn spawn_unchecked<F>(self, f: F) -> Task<F::Output>
		where
			F: Future + Send,
			F::Output: Send,
	{
		let (runnable, task) = unsafe {
			match self.target {
				Target::Any => async_task::spawn_unchecked(
					f,
					match self.priority {
						Priority::High => |r| Executor::get().push(r, Priority::High, Target::Any),
						Priority::Normal => |r| Executor::get().push(r, Priority::Normal, Target::Any),
						Priority::Low => |r| Executor::get().push(r, Priority::Low, Target::Any),
					},
				),
				Target::Thread(thread) => match self.priority {
					Priority::High => async_task::spawn_unchecked(f, move |r| Executor::get().push(r, Priority::High, Target::Thread(thread))),
					Priority::Normal => async_task::spawn_unchecked(f, move |r| Executor::get().push(r, Priority::Normal, Target::Thread(thread))),
					Priority::Low => async_task::spawn_unchecked(f, move |r| Executor::get().push(r, Priority::Low, Target::Thread(thread))),
				},
			}
		};

		runnable.schedule();

		Task(task)
	}
	
	/// # Safety
	/// 
	/// Lifetimes of captures can not be statically verified!
	pub unsafe fn spawn_local_unchecked<F>(self, f: F) -> Task<F::Output>
		where
			F: Future,
	{
		let (runnable, task) = unsafe {
			async_task::spawn_unchecked(
				f,
				match self.priority {
					Priority::High => |r| Executor::get().push(r, Priority::High, Target::Thread(ThreadId::current())),
					Priority::Normal => |r| Executor::get().push(r, Priority::Normal, Target::Thread(ThreadId::current())),
					Priority::Low => |r| Executor::get().push(r, Priority::Low, Target::Thread(ThreadId::current())),
				},
			)
		};

		runnable.schedule();

		Task(task)
	}

	#[inline(always)]
	pub fn new() -> Self {
		Self::default()
	}

	#[inline(always)]
	pub const fn priority(mut self, priority: Priority) -> Self {
		self.priority = priority;
		self
	}

	// Ignored with spawn_local.
	#[inline(always)]
	pub fn target(mut self, target: Target) -> Self {
		self.target = target;
		self
	}
}

/// # Safety
/// 
/// Do not call mem::forget on this!
#[derive(Debug)]
#[must_use = "Drop will await all tasks spawned from the scope!"]
pub struct Scope {
	count: CachePadded<AtomicUsize>,
	complete: CachePadded<AtomicUsize>,
	waker: CachePadded<Mutex<Option<Waker>>>,
	awaited: bool,
}

impl Scope {
	#[inline(always)]
	pub const fn new() -> Self {
		Self {
			count: CachePadded::new(AtomicUsize::new(0)),
			complete: CachePadded::new(AtomicUsize::new(0)),
			waker: CachePadded::new(Mutex::new(None)),
			awaited: false,
		}
	}

	pub async fn closure<F, Fut>(self, f: F) -> Fut::Output
		where
			F: FnOnce(&Self) -> Fut,
			Fut: Future,
	{
		let result = f(&self).await;
		self.await;
		result
	}
	
	#[inline(always)]
	pub fn spawn<'a, F>(&'a self, f: F) -> ScopedTask<F::Output>
		where
			F: Future + Send + 'a,
			F::Output: Send + 'a,
	{
		Builder::new().spawn_scoped(self, f)
	}
	
	#[inline(always)]
	pub fn spawn_local<'a, F>(&'a self, f: F) -> ScopedTask<F::Output>
		where
			F: Future + 'a,
	{
		Builder::new().spawn_local_scoped(self, f)
	}
	
	#[inline(always)]
	fn add_task(&self) {
		self.count.fetch_add(1, Ordering::Release);
	}
	
	/// Returns whether all tasks have completed and the scope is exiting.
	fn complete_task(&self) -> bool {
		let complete = self.complete.fetch_add(1, Ordering::AcqRel) + 1;
		
		if complete == self.count.load(Ordering::Acquire) {
			if let Some(waker) = self.waker.lock().unwrap().take() {
				waker.wake();
				return true;
			}
		}
		
		false
	}
	
	fn cancel_task(&self) {
		self.complete.fetch_sub(1, Ordering::Release);
	}
}

impl Future for Scope {
	type Output = ();
	fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
		let count = self.count.load(Ordering::Acquire);// This value should no longer change.
		
		if self.complete.load(Ordering::Acquire) == count {
			Poll::Ready(())
		} else {
			let mut waker = self.waker.lock().unwrap();
			
			// Check one more time to avoid race-conditions. Else assign waker.
			if self.complete.load(Ordering::Acquire) == count {
				drop(waker);
				self.awaited = true;

				Poll::Ready(())
			} else {
				*waker = Some(cx.waker().clone());
				
				Poll::Pending
			}
		}
	}
}

impl Drop for Scope {
	fn drop(&mut self) {
		if !self.awaited {
			Executor::get().block_on(self);
		}
	}
}

/// Task priority. Dequeued in order of priority. Normal priority tasks
/// use more efficient queues. The vast majority of tasks should be Normal
/// priority. High and Low priority use queues that waste less cycles when there
/// are no thread local tasks to dequeue or no tasks at all (should be more
/// likely with these priorities).
#[derive(Default, Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum Priority {
	High,
	#[default]
	Normal,
	Low,
}

impl Priority {
	pub const COUNT: usize = 3;
}

#[derive(Default, Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum Target {
	#[default]
	Any,
	Thread(ThreadId),
}

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum ExitReason {
	Condition,
	Shutdown,
	Empty,
}

#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub struct ThreadId(pub usize);

impl ThreadId {
	pub const MAIN: ThreadId = ThreadId(0);

	#[inline(always)]
	pub fn current() -> Self {
		local(|ThreadLocalState { id, .. }| *id)
	}
	
	#[inline(always)]
	pub fn is_main(self) -> bool {
		self == ThreadId::MAIN
	}
}

impl Debug for ThreadId {
	fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
		if self.is_main() {
			f.write_str("MAIN")
		} else {
			f.write_fmt(format_args!("WORKER[{}]", self.0 - 1))
		}
	}
}

#[inline(always)]
pub fn is_in_main_thread() -> bool {
	ThreadId::current().is_main()
}

pub struct Executor {
	pending_shutdown: AtomicBool,
	threads: Vec<ThreadState>,
	asleep_threads_mask: CachePadded<AtomicU32>,
	high_pri_queue: ArrayQueue<Runnable>,
	low_pri_queue: ArrayQueue<Runnable>,
	worker_handles: Mutex<Vec<JoinHandle<()>>>,
}

impl Executor {
	#[inline(always)]
	pub fn get() -> &'static Self {
		EXECUTOR.get().unwrap_or_else(|| Self::init())
	}
	
	/// Polls the future on the current thread to completion while still using the
	/// executor to busy-wait when the future's pending. Can capture local-state.
	pub fn block_on<'a, F>(&self, mut f: F) -> F::Output
		where
			F: Future + 'a,
	{
		struct State<'a> {
			awake: AtomicBool,
			thread: ThreadId,
			exec: &'a Executor,
		}
		
		unsafe fn wake(data: *const ()) {
			let State { awake, thread, exec } = unsafe { &*(data as *const _) };
			awake.store(true, Ordering::Release);
			exec.wake(*thread);
		}
		
		const VTABLE: RawWakerVTable = RawWakerVTable::new(
			|data| RawWaker::new(data, &VTABLE),
			wake,
			wake,
			|_| {},
		);
		
		let state = State {
			awake: AtomicBool::new(true),
			thread: ThreadId::current(),
			exec: self,
		};
		
		let waker = unsafe { Waker::new(&state as *const _ as *const _, &VTABLE) };
		let mut cx = Context::from_waker(&waker);
		loop {
			let f = unsafe { Pin::new_unchecked(&mut f) };
			if let Poll::Ready(output) = f.poll(&mut cx) {
				return output;
			}
			
			// No longer awake. Work until awoken.
			state.awake.store(false, Ordering::Release);
			
			self.work_while(|| !state.awake.load(Ordering::Acquire));
		}
	}
	
	pub fn work_while(&self, cond: impl Fn() -> bool) -> ExitReason {
		let previous_nested_calls = NESTED_CALLS.get();
		NESTED_CALLS.set(previous_nested_calls + 1);
		defer! {
			NESTED_CALLS.set(previous_nested_calls);
		}

		let mut exit = ExitReason::Condition;
		
		while cond() {
			match self.pop().map_or_else(|| self.try_wait_for_work(&cond), Ok) {
				Ok(task) => {
					task.run();
				},
				Err(out) => {
					exit = out;
					break;
				},
			};
		}
		
		exit
	}
	
	#[inline(always)]
	pub fn has_shutdown(&self) -> bool {
		self.pending_shutdown.load(Ordering::Acquire)
	}
	
	#[inline(always)]
	pub fn thread_count(&self) -> usize {
		self.threads.len()
	}
	
	#[inline(always)]
	pub fn worker_count(&self) -> usize {
		self.thread_count() - 1
	}
	
	// Rare branch taken when there are no tasks available. Most likely going to
	// put thread to sleep. Separated from work_while to not pollute instruction cache
	// and branch prediction with cold code.
	#[cold]
	#[inline(never)]
	fn try_wait_for_work(&self, cond: &impl Fn() -> bool) -> Result<Runnable, ExitReason> {
		const SPIN_COUNT: usize = 1024;
		let mut retry_count = 0;
		
		std::hint::spin_loop();
		
		loop {
			if retry_count < SPIN_COUNT {
				retry_count += 1;
				thread::yield_now();
			} else {// Conditionally put thread to sleep.
				let thread_mask = 1 << ThreadId::current().0;

				// Mark as asleep.
				let previous_asleep_threads_mask = self.asleep_threads_mask.fetch_or(thread_mask, Ordering::AcqRel);
				
				// Rare branch.
				// Try one more time to dequeue work after marking as asleep (there's potentially a race-condition between popping a job, failing,
				// and then going to sleep. Other threads can potentially fail to wake up this thread with pushed work in-between popping and marking as asleep).
				if let Some(task) = self.pop() {
					// Mark as awake.
					if self.asleep_threads_mask.fetch_and(!thread_mask, Ordering::AcqRel) & thread_mask == 0 {
						// If this is already marked as awake, then our Parker has been un-parked by another thread while we were marked as asleep. Just
						// consume the notify token in that case, otherwise the next time we really park no sleep will occur!
						local(|ThreadLocalState { parker, .. }| parker.park_timeout(Duration::ZERO));
					}
					
					return Ok(task);
				}

				// Every thread has been marked as asleep! This is the last awake thread. Conditionally begin shutdown.
				if previous_asleep_threads_mask | thread_mask == u32::MAX >> (32 - self.thread_count()) {
					if NESTED_CALLS.get() == 1 {// Shutdown. Nothing more to do.
						// Initiate shutdown.
						assert!(!self.pending_shutdown.load(Ordering::Acquire), "Shutdown already initiated!");
						
						self.pending_shutdown.store(true, Ordering::Release);
						
						if is_in_main_thread() {
							self.shutdown();
							return Err(ExitReason::Shutdown);
						} else {
							// Wake up the main thread to begin shutdown.
							let result = self.wake(ThreadId::MAIN);
							assert!(result, "Failed to wake up main thread! This should be the only thread awake?");
						}
					} else {// No more work but still executing a future so just break out and continue.
						// Just unset that bit. No other thread is awake so there's no race condition.
						self.asleep_threads_mask.fetch_and(!thread_mask, Ordering::Release);
						return Err(ExitReason::Empty);
					}
				}

				// Sleep thread until another thread awakens this.
				local(|ThreadLocalState { parker, .. }| parker.park());
				
				assert_eq!(
					self.asleep_threads_mask.load(Ordering::Acquire) & thread_mask, 0,
					"Thread un-parked but still marked as asleep! The thread un-parking should be marking the thread as awake atomically!",
				);
			}
			
			// Shutting down.
			if self.pending_shutdown.load(Ordering::Acquire) {
				if is_in_main_thread() {
					self.shutdown();
				}
				
				return Err(ExitReason::Shutdown);
			}
			
			// Exit.
			if !cond() {
				return Err(ExitReason::Condition);
			}
			
			if let Some(task) = self.pop() {
				return Ok(task);
			}
		}
	}

	fn wake(&self, thread: ThreadId) -> bool {
		// You're already awake.
		if thread == ThreadId::current() {
			return false;
		}

		let mask = 1 << thread.0;
		if self.asleep_threads_mask.load(Ordering::Acquire) & mask == 0
			|| self.asleep_threads_mask.fetch_and(!mask, Ordering::AcqRel) & mask == 0
		{
			return false;
		}

		self.threads[thread.0].unparker.unpark();
		true
	}

	/// Returns None if all threads are awake. Else returns Some with the thread that was awoken.
	fn wake_any(&self) -> Option<ThreadId> {
		let mut asleep_threads_mask = self.asleep_threads_mask.load(Ordering::Acquire);

		let thread;
		loop {
			// All threads are awake.
			if asleep_threads_mask == 0 {
				return None;
			}

			// Isolate the lowest set bit.
			let lowest_set_bit_mask = asleep_threads_mask & asleep_threads_mask.wrapping_neg();
			asleep_threads_mask = self.asleep_threads_mask.fetch_and(!lowest_set_bit_mask, Ordering::AcqRel);

			// There's the potential for a race-condition between the load and the fetch_and so we have to re-check the previous
			// value of asleep_threads_mask to ensure that the lowest_set_bit_mask is still present in the previous value. Otherwise
			// another thread raced us and woke up that thread. If so we need to retry with a different value.
			if asleep_threads_mask & lowest_set_bit_mask != 0 {
				thread = ThreadId(lowest_set_bit_mask.trailing_zeros() as usize);
				break;
			} else {
				thread::yield_now();
			}
		}

		// Wake.
		self.threads[thread.0].unparker.unpark();

		Some(thread)
	}

	/// Returns the number of threads awoken.
	fn wake_all(&self) -> usize {
		let mut asleep_threads_mask = self.asleep_threads_mask.swap(0, Ordering::AcqRel);
		let count = asleep_threads_mask.count_ones() as usize;
		
		while asleep_threads_mask != 0 {
			let thread = ThreadId(asleep_threads_mask.trailing_zeros() as usize);
			
			self.threads[thread.0].unparker.unpark();
			
			// Remove lowest set bit.
			asleep_threads_mask &= asleep_threads_mask - 1;
		}
		
		count
	}
	
	#[cold]
	#[inline(never)]
	fn shutdown(&self) {
		assert!(is_in_main_thread(), "Only the main thread can initiate shutdown!");
		
		// Mark as shutting down.
		self.pending_shutdown.store(true, Ordering::Release);
		
		let count = self.wake_all();
		assert_ne!(count, 0, "No threads to wake up?");
		
		// Join all worker threads.
		for handle in self.worker_handles.lock().unwrap().drain(..) {
			handle.join().unwrap();
		}
	}

	#[cold]
	#[inline(never)]
	fn init() -> &'static Self {
		let worker_count = env::var("WORKER_COUNT")
			.map(|v| v
				.parse::<usize>()
				.expect("WORKER_COUNT must be a number!")
			)
			.unwrap_or_else(|_| thread::available_parallelism()
				.expect("available_parallelism is unavailable!")
				.get() - 1
			);
		assert_ne!(worker_count, 0, "Unsupported at the moment!");

		let thread_count = worker_count + 1;
		assert!(thread_count <= 32, "Thread count ({thread_count}) can not exceed 32!");

		let priority_queue_capacity = env::var("PRIORITY_QUEUE_CAPACITY")
			.map(|v| v
				.parse::<usize>()
				.expect("PRIORITY_QUEUE_CAPACITY must be a number!")
			)
			.unwrap_or_else(|_| 256);

		let parkers = (0..thread_count)
			.map(|_| Parker::new())
			.collect::<Vec<_>>();

		EXECUTOR.set(Executor {
			pending_shutdown: AtomicBool::new(false),
			threads: parkers
				.iter()
				.map(|parker| ThreadState {
					unparker: parker.unparker().clone(),
					local_queues: from_fn(|_| Local::new()),
				})
				.collect(),
			asleep_threads_mask: CachePadded::new(AtomicU32::new(0)),
			high_pri_queue: ArrayQueue::new(priority_queue_capacity),
			low_pri_queue: ArrayQueue::new(priority_queue_capacity),
			worker_handles: Mutex::new(Vec::with_capacity(worker_count)),
		}).unwrap_or_else(|_| panic!("Failed to initialize executor!"));
		let executor = EXECUTOR.get().unwrap();

		let workers = (0..thread_count)
			.map(|_| Worker::new_fifo())
			.collect::<Vec<_>>();

		let stealers = workers
			.iter()
			.map(|worker| worker.stealer())
			.collect::<Vec<_>>();

		let stealers_for_thread = |id: ThreadId| stealers
			.iter()
			.enumerate()
			.filter(|(i, _)| *i != id.0)
			.map(|(_, stealer)| stealer.clone())
			.collect::<Vec<_>>();

		let mut workers = workers.into_iter().enumerate();
		let mut parkers = parkers.into_iter();

		LOCAL.with(|s| s.set(ThreadLocalState {
			id: ThreadId::MAIN,
			normal_pri_queue: workers.next().unwrap().1,
			normal_pri_stealers: stealers_for_thread(ThreadId::MAIN),
			parker: parkers.next().unwrap(),
		}).unwrap_or_else(|_| panic!("Failed to initialize thread local state!")));
		
		let mut worker_handles = executor.worker_handles.lock().unwrap();
		
		*worker_handles = workers
			.zip(parkers)
			.map(|((i, worker), parker)| {
				let id = ThreadId(i);
				let normal_pri_stealers = stealers_for_thread(id);

				thread::Builder::new()
					.name(format!("worker[{}]", i - 1))
					.spawn(move || {
						LOCAL.with(|s| s.set(ThreadLocalState {
							id,
							normal_pri_queue: worker,
							normal_pri_stealers,
							parker,
						}).unwrap_or_else(|_| panic!("Failed to initialize thread local state!")));

						executor.work_while(|| true);
					})
					.expect("Failed to spawn worker thread!")
			})
			.collect();
		
		executor
	}

	/// Pushes a task onto the Executor's pool of tasks. Occasionally waking up a thread.
	fn push(&self, task: Runnable, priority: Priority, target: Target) {
		let push_normal_pri = |task: Runnable| LOCAL.with(|s| s.get().unwrap().normal_pri_queue.push(task));
		let push_pri_queue = |queue: &ArrayQueue<Runnable>, task: Runnable| queue.push(task).unwrap_or_else(push_normal_pri);// If priority queue is full. Push to normal queue.

		match (priority, target) {
			(Priority::High, Target::Any) => push_pri_queue(&self.high_pri_queue, task),
			(Priority::Normal, Target::Any) => push_normal_pri(task),
			(Priority::Low, Target::Any) => push_pri_queue(&self.low_pri_queue, task),
			(priority, Target::Thread(thread)) => self.threads[thread.0].local_queues[priority as usize].push(task),
		}
		
		// Optionally wake up thread based on some heuristics.
		// @TODO: Implement heuristics. Don't always wake up thread.
		match target {
			Target::Any => _ = self.wake_any(),
			Target::Thread(thread) => _ = self.wake(thread),
		}
	}

	#[must_use]
	fn pop(&self) -> Option<Runnable> {
		let [high_pri_local_queue, normal_pri_local_queue, low_pri_local_queue] = &self.threads[ThreadId::current().0].local_queues;
		
		// Return Some(_) if expression results to Some(_). Else continues.
		macro_rules! return_if_some {
			($expr:expr) => {
				if let Some(task) = $expr {
					return Some(task);
				}
			}
		}
		
		// High priority.
		return_if_some!(high_pri_local_queue.pop());
		return_if_some!(self.high_pri_queue.pop());
		
		// Normal priority.
		return_if_some!(normal_pri_local_queue.pop());
		return_if_some!(LOCAL.with(|s| {
			let ThreadLocalState { normal_pri_queue, normal_pri_stealers, .. } = s.get().unwrap();
			
			return_if_some!(normal_pri_queue.pop());
			
			for stealer in normal_pri_stealers.choose_multiple(&mut rng(), normal_pri_stealers.len()) {
				loop {
					match stealer.steal() {
						Steal::Success(task) => return Some(task),
						Steal::Retry => thread::yield_now(),
						Steal::Empty => break,
					}
				}
			}
			
			None
		}));
		
		// Low priority.
		return_if_some!(low_pri_local_queue.pop());
		return_if_some!(self.low_pri_queue.pop());
		
		None
	}
}

impl Drop for Executor {
	fn drop(&mut self) {
		if !self.pending_shutdown.load(Ordering::Acquire) {
			self.shutdown();
		}
	}
}

struct ThreadState {
	unparker: Unparker,
	local_queues: [Local<Runnable>; Priority::COUNT],
}

struct ThreadLocalState {
	id: ThreadId,
	normal_pri_queue: Worker<Runnable>,
	normal_pri_stealers: Vec<Stealer<Runnable>>,
	parker: Parker,
}

struct Local<T>(CachePadded<Mutex<VecDeque<T>>>);

impl<T> Local<T> {
	const fn new() -> Self {
		Self(CachePadded::new(Mutex::new(VecDeque::new())))
	}

	fn push(&self, value: T) {
		self.0.lock().unwrap().push_back(value);
	}

	#[must_use]
	fn pop(&self) -> Option<T> {
		self.0.lock().unwrap().pop_front()
	}
}

// Convenience function for getting the ThreadLocalState.
#[inline(always)]
fn local<T>(f: impl FnOnce(&ThreadLocalState) -> T) -> T {
	LOCAL.with(|s| f(s.get().expect("ThreadLocalState is not initialized!")))
}


type Runnable = async_task::Runnable;