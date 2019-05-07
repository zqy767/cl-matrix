(in-package #:cl-matrix)

(defclass matrix-class ()
  ((m-row :initform 0 :accessor row :type fixnum)
   (n-col :initform 0 :accessor col :type fixnum)
   (element-type :initform :float :initarg :element-type :reader matrix-element-type)
   (cpu-content :accessor cpu-matrix :initform nil)
   (cpu-void :accessor cpu-void :initarg :cpu-void :initform nil)
   (gpu-content :accessor gpu-matrix :initform nil)
   (current :initform 'cpu-matrix :accessor current)
   (transpose :initform :cublas_op_n :accessor transpose)))

(defmacro matrix-elt (matrix row col)
  `(cffi:mem-aref (cpu-matrix ,matrix)
                  (find-cffi-name (matrix-element-type ,matrix))
                  (+
                   (* (row ,matrix)
                      ,col)
                   ,row)))

(defmacro alloc-cpu (matrix)
  `(when (null (cpu-void ,matrix))
     (setf (cpu-matrix ,matrix)
           (cffi:foreign-alloc (find-cffi-name (matrix-element-type ,matrix))
                               :count (* (row ,matrix) (col ,matrix)) :initial-element
                               (make-number-by-type 0 (matrix-element-type ,matrix))))
     (trivial-garbage:finalize ,matrix (lambda ()
                                         (cffi:foreign-free (cpu-matrix ,matrix))))))

(defmacro gpu* (matrix)
  `(cffi:mem-ref (gpu-matrix ,matrix) :pointer))

(defmacro alloc-gpu (matrix)
  `(when (null (gpu-matrix ,matrix))
     (setf (gpu-matrix ,matrix) (cffi:foreign-alloc :pointer))
     (cublasalloc (* (col ,matrix) (row ,matrix)) (find-size (matrix-element-type ,matrix)) (gpu-matrix ,matrix))
     (trivial-garbage:finalize ,matrix (lambda ()
                                         (cublasfree (gpu* ,matrix))))))

(defmethod set-matrix ((matrix-class matrix-class) (matrix list))
  (setf (row matrix-class) (length matrix)
        (col matrix-class) (length (car matrix)))
  (alloc-cpu matrix-class)
  (loop for j fixnum below (col matrix-class) do
       (loop for i fixnum below (row matrix-class) do
            (setf (matrix-elt matrix-class i j)
                  (make-number-by-type (nth j (nth i matrix)) (matrix-element-type matrix-class))))))

(defmethod set-matrix ((matrix-class matrix-class) (matrix simple-array))
  (typecase matrix
    ((array * (*))
     (setf
      (row matrix-class) (array-dimension matrix 0)
      (col matrix-class) (array-dimension (aref matrix 0) 0))
     (alloc-cpu matrix-class)
     (loop for j fixnum below (col matrix-class) do
          (loop for i fixnum below (row matrix-class) do
               (setf (matrix-elt matrix-class i j)
                     (make-number-by-type (aref (aref matrix i) j) (matrix-element-type matrix-class))))))
    ((array * (* *))
     (setf
      (row matrix-class) (array-dimension matrix 0)
      (col matrix-class) (array-dimension matrix 1))
     (alloc-cpu matrix-class)
     (loop for j fixnum below (col matrix-class) do
          (loop for i fixnum below (row matrix-class) do
               (setf (matrix-elt matrix-class i j)
                     (make-number-by-type (aref matrix i j) (matrix-element-type matrix-class))))))))

(defmethod set-matrix ((matrix-class matrix-class) (matrix t))
  (cond
    ((cffi:pointerp matrix)
     (assert (/= 0 (+ (row matrix-class) (col matrix-class))))
     (setf (cpu-void matrix-class) nil
           (cpu-matrix matrix-class) matrix))
    (t
     (error "this type unsupport now"))))

(defmethod initialize-instance :after ((matrix-class matrix-class) &key matrix col row  &allow-other-keys)
  (with-slots (m-row n-col) matrix-class
    (and col
         row
         (setf m-row row
               n-col col))
    (if matrix
        (set-matrix matrix-class matrix)
        (alloc-cpu matrix-class))))

(defmacro make-matrix (&key element-type matrix row col cpu-void)
  `(make-instance 'matrix-class
                  ,@(if element-type
                        `(:element-type ,element-type))
                  ,@(if matrix
                        `(:matrix ,matrix))
                  ,@(if row
                        `(:row ,row))
                  ,@(if col
                        `(:col ,col))
                  ,@(if cpu-void
                        `(:cpu-void ,cpu-void))))

(defmethod copy-matrix ((matrix matrix-class))
  (let ((ans (make-matrix :row (row matrix)
                          :col (col matrix)
                          :element-type (matrix-element-type matrix))))
    (loop for i below (* (row matrix) (col matrix)) do
         (setf (cffi:mem-aref (cpu-matrix ans) (find-cffi-name (matrix-element-type ans)) i)
               (cffi:mem-aref (cpu-matrix matrix) (find-cffi-name (matrix-element-type matrix)) i)))
    ans))

(defmethod part-copy-matrix ((matrix matrix-class) new-row new-col)
  (let ((ans (make-matrix :row new-row :col new-col :element-type (matrix-element-type matrix))))
    (loop for i below (min new-row (row matrix)) do
         (loop for j below (min new-col (col matrix)) do
              (setf (matrix-elt ans i j)
                    (matrix-elt matrix i j))))
    ans))

(defmethod transfer-cpu-to-gpu ((matrix matrix-class))
  (if (cpu-void matrix)
      (alloc-gpu matrix)
      (when (eq (current matrix) 'cpu-matrix)
        (alloc-gpu matrix)
        (with-cublas-success-ensured
            (cublassetmatrix (row matrix) (col matrix) (find-size (matrix-element-type matrix))
                             (cpu-matrix matrix) (row matrix)
                             (gpu* matrix) (row matrix)))))
  (setf (current matrix) 'gpu-matrix))

(defmacro transfer-matrixs-cpu-to-gpu (&rest matrixs)
  `(progn
     ,@(loop for matrix in matrixs collect
            `(transfer-cpu-to-gpu ,matrix))))

(defmethod transfer-gpu-to-cpu ((matrix matrix-class))
  (when (eq (current matrix) 'gpu-matrix)
    (if (cpu-void matrix)
        (setf (cpu-void matrix) nil
              (cpu-matrix matrix) (cffi:foreign-alloc (find-cffi-name (matrix-element-type matrix))
                                                      :count (* (row matrix) (col matrix)))))
    (with-cublas-success-ensured
        (cublasgetmatrix (row matrix) (col matrix) (find-size (matrix-element-type matrix))
                         (gpu* matrix) (row matrix)
                         (cpu-matrix matrix) (row matrix))))
  (setf (current matrix) 'cpu-matrix))

(defmacro transfer-matrixs-gpu-to-cpu (&rest matrixs)
  `(progn
     ,@(loop for matrix in matrixs collect
            `(transfer-gpu-to-cpu ,matrix))))

(defmethod print-object ((matrix matrix-class) stream)
  (if (null (cpu-void matrix))
      (loop for j below (col matrix) do
           (loop for i below (row matrix) do
                (format stream "~a "
                        (matrix-elt matrix i j)))
           (format stream "~%"))
      (format stream "no cpu alloc")))

(defun row-equal-check (matrix &rest matrixs)
  (assert
   (apply #'= (row matrix) (loop for other in matrixs collect (row other))))
  t)

(defun col-equal-check (matrix &rest matrixs)
  (assert
   (apply #'= (col matrix) (loop for other in matrixs collect (col other))))
  t)

(defun equal-check (matrix matrixs)
  (when (not (null matrixs))
    (assert (and
             (apply #'row-equal-check matrix matrixs)
             (apply #'col-equal-check matrix matrixs))))
  t)

(defun mutiply-check (matrixs-and-numbers)
  (loop for i in matrixs-and-numbers with m = nil do
       (if (typep i 'matrix-class)
           (if m
               (progn
                 (assert (= m (row i)))
                 (setf m (col i)))
               (setf m (col i)))
           (assert (numberp i)))
     finally (return t)))

(defmethod concat ((matrix1 matrix-class) (matrix2 matrix-class)
                   &key (direction :horizontal))
  (ccase direction
    (:horizontal
     (row-equal-check matrix1 matrix2)
     (let ((answer (make-instance 'matrix-class
                                  :row (row matrix1)
                                  :col (+ (col matrix1) (col matrix2))
                                  :element-type (matrix-element-type matrix1))))
       (loop for i below (row answer) do
            (loop for j below (col matrix1) do
                 (setf (matrix-elt answer i j) (matrix-elt matrix1 i j)))
            (loop for j below (col matrix2) do
                 (setf (matrix-elt answer i (+ (col matrix1) j)) (matrix-elt matrix2 i j))))
       answer))
    (:vertical
     (col-equal-check matrix1 matrix2)
     (let ((answer (make-instance 'matrix-class
                                  :row (+ (row matrix1) (row matrix2))
                                  :col (col matrix1)
                                  :element-type (matrix-element-type matrix1))))
       (loop for j below (col answer) do
            (loop for i below (row matrix1) do
                 (setf (matrix-elt answer i j) (matrix-elt matrix1 i j)))
            (loop for i below (row matrix2) do
                 (setf (matrix-elt answer (+ (row matrix1) i) j) (matrix-elt matrix2 i j))))
       answer))))
