# Generated by extendr: Do not edit by hand

# nolint start

#
# This file was created with the following call:
#   .Call("wrap__make_sparse_llt_wrappers", use_symbols = TRUE, package_name = "sparse.llt")

#' @docType package
#' @usage NULL
#' @useDynLib sparse.llt, .registration = TRUE
NULL

#' Perform a symbolic decomposition of invomega0 + x_invsigma_x and return a pointer to it
#' @export
beta_symbolic <- function(invomega0, x_invsigma_x) .Call(wrap__beta_symbolic, invomega0, x_invsigma_x)

#' Draw beta from N(omegabar * x_invsigma_y, omegabar)
#' @export
beta_draw <- function(invomega0, x_invsigma_x, x_invsigma_y, random_vec, symb, symb2) .Call(wrap__beta_draw, invomega0, x_invsigma_x, x_invsigma_y, random_vec, symb, symb2)

#' Calculate the symbolic union of two sparse matrices to speed up future binary operations between the two
#' @export
symbolic_union <- function(lhs, rhs) .Call(wrap__symbolic_union, lhs, rhs)


# nolint end
