use proc_macro::TokenStream;
use quote::quote;
use syn::{
    parse::{Parse, ParseStream},
    punctuated::Punctuated,
    Lit, LitInt, Token,
};

struct StaticTypeInput {
    ty: syn::Type,
    dimensions: Punctuated<LitInt, Token![,]>,
}

impl Parse for StaticTypeInput {
    fn parse(input: ParseStream<'_>) -> syn::Result<Self> {
        let ty = input.parse()?;
        input.parse::<Token![,]>()?;
        let dimensions = Punctuated::<LitInt, Token![,]>::parse_terminated(input)?;
        if dimensions.is_empty() {
            return Err(input.error("at least one array dimension is required"));
        }
        Ok(Self { ty, dimensions })
    }
}

pub(crate) fn rlst_static_array_impl(items: TokenStream) -> TokenStream {
    let args = items.to_string();
    let args: Vec<&str> = args.splitn(2, ',').map(|elem| elem.trim()).collect();
    let ty = args[0];
    let ty = if let Ok(ty) = syn::parse_str::<syn::Type>(ty) {
        ty
    } else {
        panic!("Type expected as first argument.");
    };
    let dims = if let Ok(syn::Expr::Array(expr_array)) = syn::parse_str(args[1]) {
        expr_array
    } else {
        panic!("Array of dimensions expected as second argument.");
    };

    let elems = dims.elems;
    let dims = elems
        .iter()
        .map(|elem| {
            if let syn::Expr::Lit(elem) = elem {
                match &elem.lit {
                    Lit::Int(int_lit) => int_lit.base10_parse::<usize>().unwrap(),
                    _ => {
                        panic!("Integer expected as dimensional argument.");
                    }
                }
            } else {
                panic!("Not a literal expression");
            }
        })
        .collect::<Vec<usize>>();

    let ndim = dims.len();
    let nelements: usize = dims.iter().product();

    let output = quote! { {
        let data = rlst::dense::data_container::ArrayContainer::<#ty, #nelements>::new();
        rlst::dense::array::Array::<_, #ndim>::new(rlst::dense::base_array::BaseArray::new(data, [#(#dims),*]))
    }
    };

    output.into()
}

pub(crate) fn rlst_dynamic_array_impl(items: TokenStream) -> TokenStream {
    let args = items.to_string();
    let (ty, remainder) = args
        .split_once(',')
        .expect("At least two arguments separated by a ',' are required.");

    let ty = if let Ok(ty) = syn::parse_str::<syn::Type>(ty.trim()) {
        ty
    } else {
        panic!("Type expected as first argument.");
    };

    if let Some((shape, alignment)) = remainder.split_once('|') {
        // The user provided shape and alignment

        let shape = if let Ok(expr) = syn::parse_str::<syn::Expr>(shape.trim()) {
            expr
        } else {
            panic!("Expression expected as second argument.");
        };
        let alignment = if let Ok(expr) = syn::parse_str::<syn::Expr>(alignment.trim()) {
            expr
        } else {
            panic!("Expression or literal expected as last argument.");
        };
        quote! {
            rlst::dense::array::AlignedDynArray::<#ty, _, { #alignment }>::from_shape(
                rlst::dense::macros::dynamic::aligned_shape(#shape)
            )
        }
        .into()
    } else {
        // User only provided the shape

        let shape = if let Ok(expr) = syn::parse_str::<syn::Expr>(remainder.trim()) {
            expr
        } else {
            panic!("Expression expected as second argument.");
        };

        quote! {{
            use rlst::dense::macros::dynamic::Allocate as _;
            rlst::dense::macros::dynamic::build::<#ty, _, _>(#shape, |request| {
                (&&request).allocate()
            })
        }}
        .into()
    }
}

pub(crate) fn rlst_static_type_impl(items: TokenStream) -> TokenStream {
    let StaticTypeInput { ty, dimensions } = match syn::parse(items) {
        Ok(input) => input,
        Err(error) => return error.into_compile_error().into(),
    };

    let dims = match dimensions
        .iter()
        .map(LitInt::base10_parse::<usize>)
        .collect::<syn::Result<Vec<_>>>()
    {
        Ok(dims) => dims,
        Err(error) => return error.into_compile_error().into(),
    };
    let nelem: usize = dims.iter().product();
    let ndim = dims.len();

    quote! {
        rlst::dense::array::Array<
            rlst::dense::base_array::BaseArray<
                rlst::dense::data_container::ArrayContainer<#ty, #nelem>,
                #ndim
            >,
            #ndim
        >
    }
    .into()
}
