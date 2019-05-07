(in-package #:cl-matrix)

(defmethod eigenvalue ((matrix square-matrix-class) &key (times 20))
  (transfer-cpu-to-gpu matrix)
  (loop for i below times
     with u = (make-idenitity (row matrix))
     with tmp = (copy-matrix matrix) do
       (multiple-value-bind (q r)
           (qr! tmp)
         (setf tmp (matrix-* r q)
               u (matrix-* u q)))
     finally (return
               (values
                (loop for i below (row tmp) collect (matrix-elt tmp i i))
                u))))
