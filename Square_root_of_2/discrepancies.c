#include <Python.h>
#include <numpy/arrayobject.h>

static PyObject *star_discrepancy_largest_rest(PyObject *self, PyObject *args) {
    PyArrayObject *point_set_arr;
    int n;

    if (!PyArg_ParseTuple(args, "O!i", &PyArray_Type, &point_set_arr, &n)) {
        return NULL;
    }

    double *point_set_data = (double *)PyArray_DATA(point_set_arr);

    double *x = (double *)malloc((n+2) * sizeof(double));
    double *y = (double *)malloc((n+1) * sizeof(double));
    double *xi = (double *)malloc((n+3) * sizeof(double));

    for (int i = 0; i < n; i++) {
        x[i+1] = point_set_data[2*i];     // Correct indexing for 2D numpy array
        y[i+1] = point_set_data[2*i + 1]; // Correct indexing for 2D numpy array
    }
    
    x[0] = 0.0;
    x[n+1] = 1.0;
    y[0] = 0.0;

    for (int i = 0; i < n+3; i++) {
        xi[i] = 1.0;
    }
    xi[0] = 0.0;

    double max_closed = 0;
    
    for (int j = 1; j <= n; j++) {
        int insert_here = 1;
        while (insert_here <= j && xi[insert_here] < y[j]) {
            insert_here++;
        }

        for (int k = j; k >= insert_here; k--) {
            xi[k+1] = xi[k];
        }
        xi[insert_here] = y[j];

        for (int k = 0; k <= j; k++) {
            if ((double)k / (double)n - x[j] * xi[k] > max_closed) {
                max_closed = (double)k / (double)n - x[j] * xi[k];
            }
        }
    }
    
    free(x);
    free(y);
    free(xi);
    
    return PyFloat_FromDouble(max_closed);
}

static PyMethodDef methods[] = {
    {"star_discrepancy_largest_rest", star_discrepancy_largest_rest, METH_VARARGS, "Calculate the star discrepancy"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "discrepancies",
    NULL,
    -1,
    methods
};

PyMODINIT_FUNC PyInit_discrepancies(void) {
    import_array();
    return PyModule_Create(&module);
}