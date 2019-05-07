(in-package #:cl-matrix)
(defun matrix-two-plus-in-gpu (matrix1 matrix2 &key (beta-value 1.0))
  (declare (type matrix-class matrix1 matrix2))
  (with-alpha-beta (1.0 beta-value (matrix-element-type matrix1))
    (with-cublas-handler (matrix-element-type matrix1)
      "geam"
      (transpose matrix1) (transpose matrix2)
      (row matrix1) (col matrix1)
      alpha (gpu* matrix1) (row matrix1)
      beta (gpu* matrix2) (row matrix2)
      (gpu* matrix1) (row matrix1))))

(defun matrix-two-mutiply-in-gpu (matrix1 matrix2 ans &key (alpha-value 1.0))
  (declare (type matrix-class matrix1 matrix2))
  (with-alpha-beta (alpha-value 1.0 (matrix-element-type matrix1))
    (with-cublas-handler (matrix-element-type matrix1)
      (if (member (matrix-element-type matrix1) '(:float :double))
          "gemm_v2"
          "gemm3m")
      (transpose matrix1) (transpose matrix2)
      (row matrix1) (col matrix2) (row matrix2)
      alpha (gpu* matrix1) (row matrix1)
      (gpu* matrix2) (row matrix2) beta
      (gpu* ans) (row ans))))

(defun matrix-mutiply-in-a-row (matrixs &optional (alpha-value 1.0))
  (cond
    ((atom matrixs)
     (transfer-cpu-to-gpu matrixs)
     matrixs)
    (t
     (let* ((matrix1 (matrix-mutiply-in-a-row (car matrixs)))
            (matrix2 (matrix-mutiply-in-a-row (cadr matrixs)))
            (ans (make-matrix :row (row matrix1)
                              :col (col matrix2)
                              :element-type (matrix-element-type matrix1)
                              :cpu-void t)))
       (transfer-cpu-to-gpu ans)
       (matrix-two-mutiply-in-gpu matrix1 matrix2 ans :alpha-value alpha-value)
       (setf (current ans) 'gpu-matrix)
       ans))))

(defun dp-best-match (matrixs)
  (let ((dp-array (make-array (list #1=(length matrixs) #1#)))
        (spilt-position (make-array (list #1# #1#)))
        (row-array (make-array #1# :initial-contents (mapcar #'row matrixs)))
        (col-array (make-array #1# :initial-contents (mapcar #'col matrixs))))
    (labels ((compute (i k j)
               (+ (aref dp-array i k)
                  (aref dp-array (1+ k) j)
                  (* (aref row-array i)
                     (aref col-array k)
                     (aref col-array j))))
             (spilt-list (list i j k)
               (case (length list)
                 (1 (car list))
                 (2 list)
                 (t (list (spilt-list (subseq list 0 (1+ k)) i k (aref spilt-position i (+ i k)))
                          (spilt-list (subseq list (1+ k)) (1+ k) j (aref spilt-position (+ k i 1) j)))))))
      (loop for k from 1 to (1- #1#) do
           (loop for start below (- #1# k)
              for end = (+ start k) do
                (let ((min (compute start start end)))
                  (setf
                   (aref dp-array start end) min
                   (aref spilt-position start end) 0)
                  (loop for point from 1 to (1- k)
                     for x = (compute start (+ start point) end) do
                       (when (< x min)
                         (setf min x
                               (aref dp-array start end) min
                               (aref spilt-position start end) point))))))
      (spilt-list matrixs 0 (1- #1#) (aref spilt-position 0 (1- #1#))))))


(defun matrix-mutiplus (matrix matrixs beta-value)
  (let ((ans (copy-matrix matrix)))
    (transfer-cpu-to-gpu ans)
    (loop for other in matrixs do
         (transfer-cpu-to-gpu other)
         (matrix-two-plus-in-gpu ans other :beta-value beta-value))
    (transfer-gpu-to-cpu ans)
    ans))

(defmethod matrix-+ (matrix &rest matrixs)
  (if (null matrixs)
      matrix
      (progn
        (equal-check matrix matrixs)
        (matrix-mutiplus matrix matrixs 1.0))))

(defmethod matrix-- (matrix &rest matrixs)
  (if (null matrixs)
      matrix
      (progn
        (equal-check matrix matrixs)
        (matrix-mutiplus matrix matrixs -1.0))))

(defun only-matrix-* (matrixs scale)
  (let ((ans (matrix-mutiply-in-a-row (dp-best-match matrixs) scale)))
    (transfer-gpu-to-cpu ans)
    ans))

(defun matrix-* (&rest matrixs-and-numbers)
  (mutiply-check matrixs-and-numbers)
  (let ((scale 1.0) matrixs)
    (loop for item in matrixs-and-numbers do
         (if (numberp item)
             (setf scale (* scale item))
             (push item matrixs)))
    (only-matrix-* (reverse matrixs) scale)))
