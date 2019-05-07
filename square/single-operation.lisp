(in-package #:cl-matrix)

(defmethod matrix-trace ((matrix square-matrix-class))
  (loop for i below #1=(col matrix)
     sum (matrix-elt matrix i i)))

(defmethod lu! ((matrix square-matrix-class) &key (pivot t))
  (transfer-cpu-to-gpu matrix)
  (let (info piv-g)
    (when pivot
      (setf piv-g (cffi:foreign-alloc :pointer))
      (cublasalloc #1=(row matrix) 4 piv-g))
    (with-work-and-size (work size)
        ((matrix-element-type matrix) "getrf"
         #1# #1# (gpu* matrix) #1# size)
        (matrix info piv-g)
      (setf info
            (with-cusolver-devinfo-handler (devinfo-c devinfo-g) ((cffi:mem-ref devinfo-c :int))
                                           (matrix-element-type matrix) "getrf"
                                           #1# #1# (gpu* matrix) #1# (vgpu* work)
                                           (if pivot
                                               (vgpu* piv-g)
                                               (cffi:null-pointer))))
      (transfer-gpu-to-cpu matrix))))

(defmethod lu ((matrix square-matrix-class) &key (pivot t))
  (with-matrix-operation (matrix (lu! ans :pivot pivot))))

(defmethod cholesky! ((matrix square-matrix-class) &key (lower? t))
  (transfer-cpu-to-gpu matrix)
  (let (info)
    (with-work-and-size (work size)
        ((matrix-element-type matrix) "potrf"
         (if lower?
             :cublas_fill_mode_lower
             :cublas_fill_mode_upper)
         #1=(row matrix) (gpu* matrix) #1# size)
        (matrix info)
      (setf info (with-cusolver-devinfo-handler (devinfo-c devinfo-g) ((cffi:mem-ref devinfo-c :int))
                                                (matrix-element-type matrix) "potrf"
                                                (if lower?
                                                    :cublas_fill_mode_lower
                                                    :cublas_fill_mode_upper)
                                                #1# (gpu* matrix) #1#
                                                (vgpu* work) (cffi:mem-ref size :int)))
      (transfer-gpu-to-cpu matrix))))

(defmethod cholesky ((matrix square-matrix-class) &key (lower? t))
  (with-matrix-operation (matrix (cholesky! ans :lower? lower?))))


(defmethod matrix-reverse! ((matrix square-matrix-class))
  (transfer-cpu-to-gpu matrix)
  (multiple-value-bind (matrix info pivot-g)
      (lu! matrix :pivot t)
    (assert (= info 0))
    (let (info (i (make-idenitity #1=(row matrix) :element-type (matrix-element-type matrix))))
      (transfer-cpu-to-gpu i)
      (setf info
            (with-cusolver-devinfo-handler (devinfo-c devinfo-g) ((cffi:mem-ref devinfo-c :int))
                                           (matrix-element-type matrix) "getrs"
                                           :cublas_op_n #1# #1# (gpu* matrix) #1# (vgpu* pivot-g)
                                           (gpu* i) #1#))
      (transfer-gpu-to-cpu i)
      (values i info))))

(defmethod matrix-reverse ((matrix square-matrix-class))
  (with-matrix-operation (matrix (matrix-reverse! ans))))

(defmethod det ((matrix square-matrix-class))
  (reduce #'*
          (loop for i below (row matrix) collect
               (matrix-elt (lu matrix) i i))))
