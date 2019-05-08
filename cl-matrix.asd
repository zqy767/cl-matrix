(asdf:defsystem #:cl-matrix
  :name "cl-matrix"
  :author "zhouqingyang"
  :description "matrix operation with gpu"
  :license "MIT"
  :version "0.1"
  :depends-on (#:cffi
               #:trivial-garbage)
  :serial t
  :components ((:file "package")
               (:module :driver
                        :serial t
                        :components
                        ((:file "cublas")
                         (:file "cusolver")
                         (:file "utils")
                         (:file "interface")))
               (:module :matrix
                        :serial t
                        :components
                        ((:file "basic-matrix")
                         (:file "single-operation")
                         (:file "arithmetic")))
               (:module :square
                        :serial t
                        :components
                        ((:file "square")
                         (:file "single-operation")
                         (:file "eigenvalue")))))
