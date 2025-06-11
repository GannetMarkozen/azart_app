use std::array::from_fn;
use std::cell::{Cell, RefCell};
use std::rc::Rc;
use azart::prelude::task::futures::executor::block_on;
use azart::*;
use bevy::prelude::*;
use azart::prelude::asset::{*, Asset};
use serde::{Deserialize, Serialize};

/*use azart::prelude::task::*;
use azart::prelude::task::futures::future::join_all;
use azart::prelude::task::futures::join;
use std::sync::atomic::{AtomicUsize, Ordering};
use azart::prelude::task::*;
use azart::prelude::task::futures::future::join_all;
use azart::prelude::ecs::*;*/

#[derive(Serialize, Deserialize, Debug)]
pub struct Something {
  value: u32,
  name: String,
  dependencies: Vec<AnyAsset>,
}

fn startup() {
  use azart::prelude::asset::{Asset, AnyAsset};

  let mut cache = AssetCache::new();
  cache.register::<Something>();

  {
    let asset = cache.store("assets/other_custom.asset", Something { value: 69, name: "Fortnite".to_owned(), dependencies: vec![] });
  	let asset = block_on(asset).unwrap();

    let asset = cache.store("assets/custom.asset", Something { value: 10, name: "Smitch".to_owned(), dependencies: vec![asset.into_any()] });
  	let asset = block_on(asset).unwrap();

    let result = cache.get_any("assets/other_custom.asset");
    println!("Loaded 1: {}", result.is_some());
  }

  let asset: Asset<Something> = block_on(cache.load("assets/custom.asset")).unwrap();

  println!("{asset:?}");

  let result = cache.get_any("assets/other_custom.asset");
  println!("Loaded 2: {}", result.is_some());

	/*let other_asset = block_on(azart::prelude::asset::Asset::<Something>::load("assets/custom.asset"));
	println!("Thing: {other_asset:?}");*/

  /*Asset::<Something>::register();

  {
    let (asset, _) = Asset::store("assets/other_custom.asset", Something { value: 69, name: "Fortnite".to_owned(), dependencies: None }, AssetFormat::Ron);
  	let asset = block_on(asset).unwrap();

    let (asset, _) = Asset::store("assets/custom.asset", Something { value: 10, name: "Smitch".to_owned(), dependencies: Some(asset) }, AssetFormat::Ron);
  	let asset = block_on(asset).unwrap();

    let result = AnyAsset::get("assets/other_custom.asset");
    println!("Loaded 1: {}", result.is_some());
  }

  let asset = block_on(Asset::<Something>::load("assets/custom.asset")).unwrap();

  println!("{asset:?}");

  let result = AnyAsset::get("assets/other_custom.asset");
  println!("Loaded 2: {}", result.is_some());*/

	/*let other_asset = block_on(azart::prelude::asset::Asset::<Something>::load("assets/custom.asset"));
	println!("Thing: {other_asset:?}");*/


}

fn main() {
	#[cfg(debug_assertions)]
	{
		unsafe { std::env::set_var("RUST_BACKTRACE", "full"); }
		color_eyre::install().unwrap();
	}

	App::new()
		.add_plugins(AzartPlugin)
		.add_systems(Startup, startup)
		.run();

	/*let something_else = CompId::of::<i32>();
	let something_else_else = CompId::of::<f32>();


	let now = std::time::Instant::now();

	Executor::get().block_on(async {
		let mut scope = Scope::new();

		let count = AtomicUsize::new(0);

		let task = scope.spawn_local(async {
			std::thread::sleep(std::time::Duration::from_secs(1));
			count.fetch_add(1, Ordering::Release);
		});

		let something = Ordering::Relaxed;
		let value = match something {
			Ordering::Release => 10,
			_ => 1,
		};

		println!("Count: {}", count.load(Ordering::Acquire));

		task.await;
		scope.await;

		println!("Count: {}", count.load(Ordering::Acquire));
	});

	println!("Duration: {}", now.elapsed().as_millis());
	return;

	let result = Executor::get().block_on(async {
		let start = std::time::Instant::now();

		let count = AtomicUsize::new(0);
		//Task::spawn_local(async { count += 1; }).await;

		let scope = Scope::new();

		Builder::new()
			.priority(Priority::High)
			.spawn_local_scoped(&scope, async {
				count.fetch_add(1, Ordering::Release);
			})
			.await;

		scope.await;

		println!("Count: {}", count.load(Ordering::Acquire));

		let tasks = (0..Executor::get().thread_count() + 1)
			.map(|_| Task::spawn(async move {
				println!("Begin on {:?}", ThreadId::current());
				std::thread::sleep(std::time::Duration::from_secs(3));
				println!("End on {:?}", ThreadId::current());
			}));

		join_all(tasks).await;

		println!("Hello world final on thread {:?}. Elapsed: {}!", ThreadId::current(), start.elapsed().as_millis());

		let tasks = (0..10)
			.map(|i| Task::spawn(async move {
				println!("[{i}]: Begin on {:?}", ThreadId::current());

				let tasks = (0..256)
					.map(|_| Task::spawn(async move {
						std::thread::sleep(std::time::Duration::from_micros(10))
					}));

				join_all(tasks).await;

				println!("[{i}]: End on {:?}", ThreadId::current());

				std::thread::sleep(std::time::Duration::from_micros(10))
			}));

		join_all(tasks).await;

		println!("Next final on thread {:?}. Elapsed: {}!", ThreadId::current(), start.elapsed().as_millis());

		20
	});
	*/

}
