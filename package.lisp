(cffi:define-foreign-library cublas-driver
  (t (:default "libcublas")))
(cffi:use-foreign-library cublas-driver)

(cffi:define-foreign-library cusolver-driver
  (t (:default "libcusolver")))
(cffi:use-foreign-library cusolver-driver)

(defpackage #:cl-matrix
  (:use #:cl #:cffi)
  (:import-from #:trivial-garbage #:finalize)
  (:export

   #:make-matrix
   #:matrix-elt
   #:copy-matrix
   #:part-copy-matrix
   #:concat
   #:matrix-transpose
   #:svd-jacbi
   #:svd-jacbi!
   #:qr
   #:qr!
   #:matrix-+
   #:matrix--
   #:matrix-*

   #:make-square
   #:make-identity
   #:lu
   #:lu!
   #:cholesky
   #:cholesky!
   #:matrix-trace
   #:matrix-reverse
   #:matrix-reverse!
   #:det
   #:eigenvalue))
