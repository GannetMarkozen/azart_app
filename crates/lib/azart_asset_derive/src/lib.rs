use proc_macro::TokenStream;
use syn::{parse_macro_input, Data, DeriveInput, Fields, Type};

#[proc_macro_derive(Asset)]
pub fn derive_asset(ts: TokenStream) -> TokenStream {
  let mut ast: DeriveInput = parse_macro_input!(ts);

  fn visit_struct(ast: &mut DeriveInput) {
    let Data::Struct(s) = &mut ast.data else {
      return;
    };

    for field in s.fields.iter_mut() {

    }
  }
}

fn is_asset(ty: &Type) -> bool {
  matches!(ty, Type::Path(p) if p.path.segments.last().map_or(false, |seg| seg.ident == "Asset"))
}
