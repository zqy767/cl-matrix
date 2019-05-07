(in-package :cl-matrix)

(defclass square-matrix-class (matrix-class)
  ())

(defmacro make-square (&key matrix n element-type cpu-void)
  `(make-instance 'square-matrix-class
                  ,@(if element-type
                        `(:element-type ,element-type))
                  ,@(if matrix
                        `(:matrix ,matrix))
                  ,@(if n
                        `(:row ,n :col ,n))
                  ,@(if cpu-void
                        `(:vpu-void ,cpu-void))))

(defun make-idenitity (n &key (element-type :float) (cpu-void nil))
  (let ((array (cffi:foreign-alloc (find-cffi-name element-type)
                                   :count (* n n) :initial-element (make-number-by-type 0 element-type))))
    (loop for i below n do
         (setf (cffi:mem-aref array (find-cffi-name element-type) (+ i (* i n)))
               (make-number-by-type 1 element-type)))
    (make-square :matrix array :n n :element-type element-type :cpu-void cpu-void)))


(defmethod copy-matrix ((matrix square-matrix-class))
  (let ((ans (call-next-method)))
    (change-class ans 'square-matrix-class)
    ans))
