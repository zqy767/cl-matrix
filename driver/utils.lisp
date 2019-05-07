(in-package #:cl-matrix)

(defmacro with-success-ensured (success-condition operate &rest condition-case)
  `(ecase ,operate
     (,success-condition
      t)
     ,@(if condition-case
           `(,@condition-case))))

(defmacro with-cublas-success-ensured (cublas-operate &rest condition-case)
  `(with-success-ensured (t :cublas_status_success) ,cublas-operate
                         ,@condition-case))

(defmacro with-cusolver-success-ensured (cusolver-operate &rest condition-case)
  `(with-success-ensured (t :cusolver_status_success) ,cusolver-operate
                         ,@condition-case))

(cffi:defcstruct (cucomplex :class :cucomplex :size 8)
    (real :float :offset 0)
  (imag :float :offset 4))

(cffi:defcstruct (cudoublecomplex :class :cudoublecomplex :size 16)
    (real :double :offset 0)
  (imag :double :offset 8))

(defmethod translate-into-foreign-memory (complex (type :cucomplex) cudata)
  (declare (type (complex complex)))
  (cffi:with-foreign-slots ((real imag) cudata (:struct cucomplex))
    (setf real (realpart complex)
          imag (imagpart complex))))

(defmethod translate-into-foreign-memory (complex (type :cudoublecomplex) cudata)
  (declare (type (complex complex)))
  (cffi:with-foreign-slots ((real imag) cudata (:struct cudoublecomplex))
    (setf real (realpart complex)
          imag (imagpart complex))))

(defmethod translate-from-foreign (cudata (type :cucomplex))
  (cffi:with-foreign-slots ((real imag) cudata (:struct cucomplex))
    (complex real imag)))

(defmethod translate-from-foreign (cudata (type :cudoublecomplex))
  (cffi:with-foreign-slots ((real imag) cudata (:struct cudoublecomplex))
    (complex real imag)))

(deftype cudata-type ()
  `(member :float
           :double
           :cucomplex
           :cudoublecomplex))

(defmacro vgpu* (var)
  `(cffi:mem-ref ,var :pointer))
