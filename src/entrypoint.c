// We need to forward routine registration from C to Rust
// to avoid the linker removing the static library.

void R_init_sparse_llt_extendr(void *dll);

void R_init_sparse_llt(void *dll) {
    R_init_sparse_llt_extendr(dll);
}
