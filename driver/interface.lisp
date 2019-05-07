(in-package #:cl-matrix)

(eval-when (:compile-toplevel :load-toplevel :execute)
  (defparameter *cublas-handler* (cffi:foreign-alloc :pointer))
  (with-cublas-success-ensured
      (cublascreate_v2 *cublas-handler*))
  (trivial-garbage:finalize *cublas-handler* (lambda ()
                                               (cublasdestroy_v2 *cublas-handler*)))

  (defparameter *cusolver-handler* (cffi:foreign-alloc :pointer))
  (with-cusolver-success-ensured
      (cusolverdncreate *cusolver-handler*))
  (trivial-garbage:finalize *cublas-handler* (lambda ()
                                               (cusolverdndestroy *cusolver-handler*)))
  (defparameter *cublas-register*
    '((:float :float 4 "S")
      (:double :double 8 "D")
      (:cucomplex (:struct cucomplex) 8 "C")
      (:cudoublecomplex (:struct cudoublecomplex) 16 "Z"))))

(defmacro find-cffi-name (cudata-type)
  `(cadr (find ,cudata-type ',*cublas-register* :key #'car)))

(defmacro find-size (cudata-type)
  `(if #1=(find ,cudata-type ',*cublas-register* :key #'car)
       (caddr #1#)
       (cffi:foreign-type-size ,cudata-type)))

(defmacro find-sepcific-letter (cudata-type)
  `(cadddr (find ,cudata-type ',*cublas-register* :key #'car)))

(defmacro single-or-double? (cudata-type)
  `(ecase ,cudata-type
     ((:float :cucomplex) :float)
     ((:double :cudoublecomplex) :double)))

(defmacro single-or-double?-with-lisp-type (cudata-type)
  `(ecase ,cudata-type
     ((:float :cucomplex) 'single-float)
     ((:double :cudoublecomplex) 'double-float)))

(defmacro make-complex (number type)
  `(complex (if (complexp ,number)
                (coerce (realpart ,number) (single-or-double?-with-lisp-type ,type))
                (coerce ,number (single-or-double?-with-lisp-type ,type)))
            (if (complexp ,number)
                (coerce (imagpart ,number) (single-or-double?-with-lisp-type ,type))
                (coerce 0 (single-or-double?-with-lisp-type ,type)))))

(defun make-number-by-type (number cudata-type)
  (ecase cudata-type
    (:float (coerce number 'single-float))
    (:double (coerce number 'double-float))
    ((:cucomplex :cudoublecomplex) (make-complex number cudata-type))))

(defmacro make-cuda-name (prefix type name)
  `(intern (concatenate 'string (string-upcase ,prefix)
                        (find-sepcific-letter ,type) (string-upcase ,name))))

(defun make-cublas-function-name (type name)
  (make-cuda-name "CUBLAS" type name))

(defun make-cusolver-function-name (type name)
  (make-cuda-name "CUSOLVERDN" type name))

(defmacro with-cublas-handler (type special-name &rest parameter)
  `(with-cublas-success-ensured
       (funcall (make-cublas-function-name ,type ,special-name)
                (cffi:mem-ref *cublas-handler* :pointer)
                ,@parameter)))

(defmacro with-cusolver-handler (type special-name &rest parameter)
  `(with-cusolver-success-ensured
       (funcall (make-cusolver-function-name ,type ,special-name)
                (cffi:mem-ref *cusolver-handler* :pointer)
                ,@parameter)))

(defmacro with-foreign-pointers (pointers &body body)
  `(cffi:with-foreign-objects ,(loop for pointer in pointers
                                  collect `(,(car pointer) ,(cadr pointer)))
     (setf ,@(loop for pointer in pointers
                collect `(cffi:mem-ref ,(car pointer) ,(cadr pointer))
                collect (caddr pointer)))
     ,@body))

(defmacro with-alpha-beta ((alpha-value beta-value type) &body body)
  `(with-foreign-pointers ((alpha (find-cffi-name ,type) (make-number-by-type ,alpha-value ,type))
                           (beta (find-cffi-name ,type) (make-number-by-type ,beta-value ,type)))
     ,@body))

;;; (cpu-name gpu-name n type)
(defmacro with-cpu-gpu-malloc ((&rest var-list) (&rest return-value) &body body)
  `(let ,(loop for var in var-list
            if (car var)
            collect `(,(car var) (cffi:foreign-alloc ,(cadddr var) :count ,(caddr var)))
            collect `(,(cadr var) (cffi:foreign-alloc :pointer)))
     ,@(loop for var in var-list
          collect `(cublasalloc ,(caddr var) (find-size ,(cadddr var)) ,(cadr var)))
     ,@body
     ,@(loop for var in var-list
          if (car var) collect `(cublasgetvector ,(caddr var) (find-size ,(cadddr var))
                                                 (cffi:mem-ref ,(cadr var) :pointer) 1 ,(car var) 1))
     ,@(loop for var in var-list
          collect `(cublasfree (vgpu* ,(cadr var))))
     ,(if return-value
          `(values ,@return-value))))

(defmacro with-work-and-size ((work size)
                              (type special-name &rest parameter) (&rest return-value) &body body)
  `(cffi:with-foreign-object (,size :int)
     (with-cusolver-handler
         ,type (concatenate 'string ,special-name "_BUFFERSIZE") ,@parameter)
     (with-cpu-gpu-malloc ((nil ,work (cffi:mem-ref ,size :int) ,type))
       ,return-value
       ,@body)))

(defmacro with-cusolver-devinfo-handler ((devinfo-c devinfo-g) (&rest return-value) type special-name &rest parameter)
  `(let ((,devinfo-c (cffi:foreign-alloc :int))
         (,devinfo-g (cffi:foreign-alloc :pointer)))
     (cublasalloc 1 4 ,devinfo-g)
     (with-cusolver-success-ensured
         (funcall (make-cusolver-function-name ,type ,special-name)
                  (cffi:mem-ref *cusolver-handler* :pointer)
                  ,@parameter
                  (vgpu* ,devinfo-g)))
     (cublasgetvector 1 4 (cffi:mem-ref ,devinfo-g :pointer) 1 ,devinfo-c 1)
     (cublasfree (vgpu* ,devinfo-g))
     (let ((tmp-dev (cffi:mem-ref ,devinfo-c :int)))
       (and (< tmp-dev 0)
            (error "the ~a parameter is wrong" (abs tmp-dev))))
     ,(if return-value
          `(values ,@return-value))))

(defmacro with-matrix-operation ((matrix matrix-op))
  `(let ((ans (copy-matrix ,matrix)))
     ,matrix-op))
