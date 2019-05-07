(in-package #:cl-matrix)

(defmethod matrix-transpose ((matrix matrix-class) &optional (transa :cublas_op_t))
  (transfer-cpu-to-gpu matrix)
  (setf (transpose matrix) :cublas_op_n)
  (let ((ans (make-matrix :row (col matrix) :col (row matrix)
                          :element-type (matrix-element-type matrix))))
    (transfer-cpu-to-gpu ans)
    (with-alpha-beta (1.0 0.0 (matrix-element-type matrix))
      (with-cublas-handler
          (matrix-element-type matrix) "geam"
          transa :cublas_op_n (row ans) (col ans)
          alpha (gpu* matrix) (row matrix)
          beta (cffi:null-pointer) (max (row matrix) (col matrix))
          (gpu* ans) (row ans)))
    (transfer-gpu-to-cpu ans)
    ans))

(defmethod svd-jacobi! ((matrix matrix-class) &key (tolerance 1d-7) (max-sweep 15))
  (declare (optimize debug))
  (let ((info (cffi:foreign-alloc :pointer))
        (u-matrix (make-matrix :row (row matrix) :col (row matrix) :element-type (matrix-element-type matrix)))
        (v-matrix (make-matrix :row (col matrix) :col (col matrix) :element-type (matrix-element-type matrix)))
        (s (make-matrix :row 1 :col (min (col matrix) (row matrix)) :element-type (single-or-double?
                                                                                   (matrix-element-type matrix)))))
    (transfer-matrixs-cpu-to-gpu matrix u-matrix v-matrix s)
    (with-cusolver-success-ensured
        (cusolverdncreategesvdjinfo info))
    (with-cusolver-success-ensured
        (cusolverdnxgesvdjsettolerance (vgpu* info) tolerance))
    (with-cusolver-success-ensured
        (cusolverdnxgesvdjsetmaxsweeps (vgpu* info) max-sweep))
    (with-work-and-size (work size)
        ((matrix-element-type matrix) "gesvdj"
         :cusolver_eig_mode_vector 0 (row matrix) (col matrix)
         (gpu* matrix) (row matrix)
         (gpu* s)
         (gpu* u-matrix) (row matrix)
         (gpu* v-matrix) (col matrix) size
         (cffi:mem-ref info :pointer)) ()
      (with-cpu-gpu-malloc ((nil status 1 :int)) ()
        (with-cusolver-handler (matrix-element-type matrix) "gesvdj"
                               :cusolver_eig_mode_vector 0 (row matrix) (col matrix)
                               (gpu* matrix) (row matrix)
                               (gpu* s)
                               (gpu* u-matrix) (row matrix)
                               (gpu* v-matrix) (col matrix)
                               (vgpu* work) (cffi:mem-ref size :int) (vgpu* status)
                               (vgpu* info))))
    (with-cusolver-success-ensured
        (cusolverdncreategesvdjinfo info))
    (transfer-matrixs-gpu-to-cpu s u-matrix v-matrix)
    (values s u-matrix v-matrix)))

(defmethod svd-jacobi ((matrix matrix-class) &key (tolerance 1d-7) (max-sweep 15))
  (with-matrix-operation (matrix (svd-jacobi! ans :tolerance tolerance :max-sweep max-sweep))))

(defmethod qr! ((matrix matrix-class))
  (declare (optimize debug))
  (transfer-cpu-to-gpu matrix)
  (let ((q-matrix (make-matrix :row (row matrix) :col (col matrix) :element-type (matrix-element-type matrix)))
        (r-matrix (make-matrix :row (row matrix) :col (col matrix) :element-type (matrix-element-type matrix))))
    (cffi:with-foreign-objects ((size1 :int) (size2 :int))
      (with-cpu-gpu-malloc ((nil tau (min (row matrix) (col matrix)) (matrix-element-type matrix))) (q-matrix r-matrix)
        (with-cusolver-handler (matrix-element-type matrix)  "geqrf_buffersize"
                               #1=(row matrix) (col matrix) (gpu* matrix) #1# size1)
        (with-cusolver-handler (matrix-element-type matrix) (if (member (matrix-element-type matrix)
                                                                        '(:float :double))
                                                                "orgqr_buffersize"
                                                                "ungqr_buffersize")
                               #1# (col matrix) (col matrix) (gpu* matrix) #1# (vgpu* tau) size2)
        (let ((size (max (cffi:mem-ref size1 :int) (cffi:mem-ref size2 :int))))
          (with-cpu-gpu-malloc ((nil work size (matrix-element-type matrix))
                                (devinfo-c devinfo-g 1 :int)) ()
            (with-cusolver-handler
                (matrix-element-type matrix) "geqrf"
                #1# (col matrix) (gpu* matrix) #1#
                (vgpu* tau) (vgpu* work) size
                (vgpu* devinfo-g))
            (cublasgetmatrix #1# (col matrix) (find-size (matrix-element-type matrix))
                             (gpu* matrix) #1# (cpu-matrix r-matrix) #1#)
            (loop for i below #1# do
                 (loop for j from 0 to (1- i) do
                      (setf (matrix-elt r-matrix i j) (make-number-by-type 0 (matrix-element-type r-matrix)))))
            (with-cusolver-handler
                (matrix-element-type matrix) (if (member (matrix-element-type matrix)
                                                         '(:float :double))
                                                 "orgqr"
                                                 "ungqr")
                #1# (col matrix) (col matrix) (gpu* matrix) #1# (vgpu* tau) (vgpu* work)
                size (vgpu* devinfo-g))
            (cublasgetmatrix #1# (col matrix) (find-size (matrix-element-type matrix))
                             (gpu* matrix) #1# (cpu-matrix q-matrix) #1#)
            (setf (current matrix) 'cpu-matrix)))))))

(defmethod qr ((matrix matrix-class))
  (with-matrix-operation (matrix (qr! ans))))
