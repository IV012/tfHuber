#include <Eigen/Dense>
#include <Python.h>
#include <cmath>

using namespace std;
using namespace Eigen;

double f1(const double x, const VectorXd sqres, const int n) {
    double sum = 0;
    for (int i = 0; i < sqres.size(); i++) {
        if (sqres(i) < x) {
            sum += sqres(i);
        }
        else
            sum += x;
    }
    return sum / n / x - log(n) / n;
}

double rootf1(const VectorXd sqres, const int n, double low, double up, const double tol = 1e-5, const int max_iter = 500) {
    int iter = 0;
    double mid, val;
    while (iter <= max_iter && (up - low) > tol) {
        mid = (up + low) / 2;
        val = f1(mid, sqres, n);
        if (val == 0) {
            return mid;
        }
        else if (val < 0) {
            up = mid;
        }
        else
            low = mid;
        iter += 1;
    }
    return (low + up) / 2;
}

double f2(const double x, const VectorXd sqres, const int n, const int d, const double z) {
    double sum = 0;
    for (int i = 0; i < sqres.size(); i++) {
        if (sqres(i) < x) {
            sum += sqres(i);
        }
        else
            sum += x;
    }
    return sum / n / x - (z + d) / n;
}

double rootf2(const VectorXd sqres, const int n, const int d, const double z, double low, double up, const double tol = 1e-5, const int max_iter = 500) {
    int iter = 0;
    double mid, val;
    while (iter <= max_iter && (up - low) > tol) {
        mid = (up + low) / 2;
        val = f2(mid, sqres, n, d, z);
        if (val == 0) {
            return mid;
        }
        else if (val < 0) {
            up = mid;
        }
        else
            low = mid;
        iter += 1;
    }
    return (low + up) / 2;
}

double f3(const double x, const VectorXd sqres, const int n, const int d) {
    int N = n * (n - 1) >> 1;
    double sum = 0;
    for (int i = 0; i < sqres.size(); i++) {
        if (sqres(i) < x) {
            sum += sqres(i);
        }
        else
            sum += x;
    }
    return sum / N / x - (2 * log(d) + log(n)) / n;

}

double rootf3(const VectorXd sqres, const int n, const int d, double low, double up, const double tol = 1e-5, const int max_iter = 500) {
    int iter = 0;
    double mid, val;
    while (iter <= max_iter && (up - low) > tol) {
        mid = (up + low) / 2;
        val = f3(mid, sqres, n, d);
        if (val == 0) {
            return mid;
        }
        else if (val < 0) {
            up = mid;
        }
        else
            low = mid;
        iter += 1;
    }
    return (low + up) / 2;
}

struct Mean {
    int iter;
    double mu;
    double tau;
    Mean() : iter(0), mu(0), tau(0) {}
};

Mean Huber_mean(const VectorXd x, const double tol = 1e-5, const int max_iter = 500) {
    double muold = 0;
    double tauold = 0;
    int iter = 0;
    int n = (int)x.size();

    VectorXd sqres(n);
    VectorXd w(n);

    double munew = x.mean();
    double taunew = 0;
    for (int i = 0; i < (int)x.size(); i++) {
        taunew += (x(i) - munew) * (x(i) - munew);
    }
    taunew = sqrt(taunew / (double(x.size()) - 1)) * sqrt((long double)n / log(n));

    while ((abs(munew - muold) > tol || abs(taunew - tauold) > tol) && iter < max_iter) {
        muold = munew;
        tauold = taunew;
        for (int i = 0; i < n; i++) {
            sqres(i) = (x(i) - muold) * (x(i) - muold);
        }
        taunew = sqrt((long double)rootf1(sqres, n, sqres.minCoeff(), sqres.sum()));
        munew = 0;
        for (int i = 0; i < n; i++) {
            w(i) = min(taunew / sqrt(sqres(i)), 1.0);
            munew += w(i) * x(i);
        }
        munew = munew / w.sum();
        iter++;
    }

    Mean List;
    List.iter = iter;
    List.mu = munew;
    List.tau = taunew;
    return List;
}

double hMeanCov(const VectorXd Z, const int n, const int d, const int N, const double tol = 1e-5, const int max_iter = 500) {
    double muold = 0;
    double munew = Z.mean();
    double tauold = 0;
    double taunew = 0;
    for (int i = 0; i < (int)Z.size(); i++) {
        taunew += (Z(i) - munew) * (Z(i) - munew);
    }
    taunew = sqrt(taunew / (double(Z.size()) - 1)) * sqrt((long double)n / (2 * log(d) + log(n)));
    int iter = 0;
    VectorXd sqres(N), w(N);

    while ((abs(munew - muold) > tol || abs(taunew - tauold) > tol) && iter < max_iter) {
        muold = munew;
        tauold = taunew;
        for (int i = 0; i < N; i++) {
            sqres(i) = (Z(i) - muold) * (Z(i) - muold);
        }
        taunew = sqrt((long double)rootf3(sqres, n, d, sqres.minCoeff(), sqres.sum()));
        munew = 0;
        for (int i = 0; i < N; i++) {
            w(i) = min(taunew / sqrt(sqres(i)), 1.0);
            munew += w(i) * Z(i);
        }
        munew = munew / w.sum();
        iter++;
    }
    return munew;
}

MatrixXd Huber_cov(const MatrixXd X, const double tol = 1e-5, const int max_iter = 500) {
    int n = (int)X.rows();
    int p = (int)X.cols();
    VectorXd mu(p);
    MatrixXd sigma(p, p);

    for (int j = 0; j < p; j++) {
        mu(j) = Huber_mean(X.col(j)).mu;
        double theta = Huber_mean(X.col(j).array().square().matrix()).mu;
        double temp = mu(j) * mu(j);
        if (theta > temp) {
            theta -= temp;
        }
        sigma(j, j) = theta;
    }
    int N = n * (n - 1) >> 1;
    MatrixXd Y(N, p);
    for (int i = 0, k = 0; i < n - 1; i++) {
        for (int j = i + 1; j < n; j++) {
            Y.row(k++) = X.row(i) - X.row(j);
        }
    }
    for (int i = 0; i < p - 1; i++) {
        for (int j = i + 1; j < p; j++) {
            sigma(i, j) = sigma(j, i) = hMeanCov(Y.col(i).cwiseProduct(Y.col(j)) / 2, n, p, N);
        }
    }
    return sigma;
}

struct Reg {
    int iter;
    VectorXd theta;
    double tau;

    Reg(int n) {
        iter = 0;
        tau = 0;
        theta = VectorXd::Zero(n);
    }
};

Reg one_step_fit(const MatrixXd X, const VectorXd Y, double z = 0, const double tol = 1e-5, const int max_iter = 500) {
    int n = (int)X.rows();
    int d = (int)X.cols();
    MatrixXd Z(n, d + 1);
    Z.block(0, 0, n, 1) = VectorXd::Ones(n);
    Z.block(0, 1, n, d) = X;


    if (z == 0) {
        z = log(n);
    }

    VectorXd thetaold = VectorXd::Zero(d + 1);
    VectorXd thetanew(d + 1);
    thetanew = (Z.transpose() * Z).inverse() * (Z.transpose() * Y);

    double sigma;
    VectorXd sqres(n);
    sqres = Y - Z * thetanew;
    for (int i = 0; i < n; i++) {
        sqres(i) *= sqres(i);
    }
    sigma = sqrt(sqres.sum() / n);

    double tauold = 0;
    double taunew = sigma * sqrt((d + z) / n);
    int iter = 0;

    MatrixXd WZ(n, d + 1);
    VectorXd WY(n);

    double w;

    while ((abs(taunew - tauold) > tol || (thetanew - thetaold).lpNorm<Infinity>() > tol) && iter < max_iter) {
        thetaold = thetanew;
        tauold = taunew;
        sqres = Y - Z * thetanew;
        for (int i = 0; i < n; i++) {
            sqres(i) *= sqres(i);
        }
        taunew = sqrt(rootf2(sqres, n, d, z, sqres.minCoeff(), sqres.sum()));
        WZ = Z;
        WY = Y;
        for (int i = 0; i < n; i++) {
            w = taunew / sqrt(sqres(i));
            if (w < 1) {
                WZ.block(i, 0, 1, d + 1) *= w;
                WY(i) *= w;
            }
        }
        thetanew = (Z.transpose() * WZ).inverse() * (Z.transpose() * WY);
        iter++;
    }
    Reg List(d + 1);
    List.iter = iter;
    List.tau = taunew;
    List.theta = thetanew;
    return List;
}

int partition(VectorXd& array, int left, int right) {
    int pos = right;
    right--;
    double temp;
    while (left <= right)
    {
        while (left < pos && array(left) <= array(pos))
            left++;

        while (right >= 0 && array(right) > array(pos))
            right--;

        if (left >= right)
            break;

        temp = array(left);
        array(left) = array(right);
        array(right) = temp;
    }
    temp = array(pos);
    array(pos) = array(left);
    array(left) = temp;

    return left;
}

double getMidValue(const VectorXd arr)
{
    if (arr.size() <= 0)
        return -1;

    VectorXd series = arr;
    int left = 0;
    int right = (int)series.size() - 1;
    int index = -1;
    int midPos = right >> 1;

    while (index != midPos)
    {
        index = partition(series, left, right);

        if (index < midPos)
        {
            left = index + 1;
        }
        else if (index > midPos)
        {
            right = index - 1;
        }
        else
        {
            break;
        }
    }
    assert(index == midPos);
    if ((int)series.size() % 2 == 1) {
        return series(index);
    }
    else {
        midPos++;
        double left_mid = series(index);
        left = index + 1;
        if (right < midPos) {
            right++;
        }
        while (index != midPos)
        {
            index = partition(series, left, right);

            if (index < midPos)
            {
                left = index + 1;
            }
            else if (index > midPos)
            {
                right = index - 1;
            }
            else
            {
                break;
            }
        }
        assert(index == midPos);

        return (series(index) + left_mid) / 2;
    }
}

Reg two_step_fit(const MatrixXd X, const VectorXd Y, const double tol = 1e-5, const double constTau = 1.345, const int max_iter = 500) {
    int n = (int)X.rows();
    int d = (int)X.cols();
    MatrixXd Z(n, d + 1);
    Z.block(0, 0, n, 1) = VectorXd::Ones(n);
    Z.block(0, 1, n, d) = X;

    VectorXd thetaold = VectorXd::Zero(d + 1);
    VectorXd thetanew(d + 1);
    thetanew = (Z.transpose() * Z).inverse() * (Z.transpose() * Y);

    double sigma;
    VectorXd sqres(n);
    sqres = Y - Z * thetanew;
    for (int i = 0; i < n; i++) {
        sqres(i) *= sqres(i);
    }
    sigma = sqrt((long double)sqres.sum() / ((long double)n - (long double)d));

    double tauold = 0;
    double taunew = sigma * sqrt((long double)n / log((long double)(d + log(n * d))));
    int iter = 0;

    MatrixXd WZ(n, d + 1);
    VectorXd WY(n);
    VectorXd res(n);
    double w, mad, mid;

    while ((abs(taunew - tauold) > tol || (thetanew - thetaold).lpNorm<Infinity>() > tol) && iter < max_iter) {
        thetaold = thetanew;
        tauold = taunew;
        res = Y - Z * thetaold;
        mid = getMidValue(res);
        mad = getMidValue((res.array() - mid).cwiseAbs().matrix()) / 0.6744898;
        taunew = constTau * mad;
        WZ = Z;
        WY = Y;
        for (int i = 0; i < n; i++) {
            w = taunew / abs(res(i));
            if (w < 1) {
                WZ.block(i, 0, 1, d + 1) *= w;
                WY(i) *= w;
            }
        }
        thetanew = (Z.transpose() * WZ).inverse() * (Z.transpose() * WY);
        iter++;
    }
    thetanew(0) = Huber_mean(Y - X * thetanew.block(1, 0, d, 1)).mu;
    Reg List(d + 1);
    List.iter = iter;
    List.tau = taunew;
    List.theta = thetanew;
    return List;
}

static PyObject* mean(PyObject*, PyObject* args, PyObject* kw) {
    double tol = 1e-5;
    int max_iter = 500;
    int shape;
    PyObject* q, * s;

    static const char* kwlist[] = { "X", "tol", "max_iter", NULL };

    if (!PyArg_ParseTupleAndKeywords(args, kw, "O|di", const_cast<char**>(kwlist), &q, &tol, &max_iter)) {
        return NULL;
    }

    // Parsing Vector
    shape = (int) PyList_Size(q);

    VectorXd X = VectorXd::Ones(shape);
    for (int i = 0; i < shape; i++) {
        s = PyList_GetItem(q, i);
        X(i) = PyFloat_AsDouble(s);
    }

    // Huber Mean
    Mean List;
    List = Huber_mean(X, tol, max_iter);
    double mu = List.mu;
    int iter = List.iter;
    double tau = List.tau;

    PyObject* result;
    result = PyTuple_New(3);

    PyTuple_SetItem(result, 0, PyFloat_FromDouble(mu));
    PyTuple_SetItem(result, 1, PyFloat_FromDouble(tau));
    PyTuple_SetItem(result, 2, Py_BuildValue("i", iter));

    return result;
}

static PyObject* cov(PyObject*, PyObject* args, PyObject* kw) {
    double tol = 1e-5;
    int max_iter = 500;
    int shape[2];
    PyObject* p, * r, * s;

    static const char* kwlist[] = { "X", "tol", "max_iter", NULL };

    if (!PyArg_ParseTupleAndKeywords(args, kw, "O|di", const_cast<char**>(kwlist), &p, &tol, &max_iter)) {
        return NULL;
    }

    // Parsing Matrix
    shape[0] = (int) PyList_Size(p);
    r = PyList_GetItem(p, 0);
    MatrixXd X;
    if (PyList_Check(r)) {
        shape[1] = (int)PyList_Size(r);

        X = MatrixXd::Ones(shape[0], shape[1]);
        for (int i = 0; i < shape[0]; i++) {
            for (int j = 0; j < shape[1]; j++) {
                r = PyList_GetItem(p, i);
                s = PyList_GetItem(r, j);
                X(i, j) = PyFloat_AsDouble(s);
            }
        }
    }
    else if (PyFloat_Check(r)) {
        shape[1] = 1;

        X = MatrixXd::Ones(shape[0], shape[1]);
        for (int i = 0; i < shape[0]; i++) {
            s = PyList_GetItem(p, i);
            X(i, 0) = PyFloat_AsDouble(s);
        }
    }
    else return 0;
        

    // Huber Covariance
    MatrixXd COV;
    COV = Huber_cov(X, tol, max_iter);


    PyObject* result;
    result = PyList_New(0);
    for (int i = 0; i < COV.rows(); i++) {
        PyObject* coef;
        coef = PyList_New(0);
        for (int j = 0; j < COV.cols(); j++) {
            PyList_Append(coef, PyFloat_FromDouble(COV(i, j)));
        }
        PyList_Append(result, coef);
    }

    return result;
}

static PyObject* one_step(PyObject*, PyObject* args, PyObject* kw) {
    double z = 0, tol = 1e-5;
    int max_iter = 500;
    int shape[2];
    PyObject* p, * q, * r, * s;

    static const char* kwlist[] = { "X", "Y", "z", "tol", "max_iter", NULL };

    if (!PyArg_ParseTupleAndKeywords(args, kw, "OO|ddi", const_cast<char**>(kwlist), &p, &q, &z, &tol, &max_iter)) {
        return NULL;
    }

    // Parsing X
    shape[0] = (int) PyList_Size(p);
    r = PyList_GetItem(p, 0);
    MatrixXd X;
    if (PyList_Check(r)) {
        shape[1] = (int)PyList_Size(r);

        X = MatrixXd::Ones(shape[0], shape[1]);
        for (int i = 0; i < shape[0]; i++) {
            for (int j = 0; j < shape[1]; j++) {
                r = PyList_GetItem(p, i);
                s = PyList_GetItem(r, j);
                X(i, j) = PyFloat_AsDouble(s);
            }
        }
    }
    else if (PyFloat_Check(r)) {
        shape[1] = 1;

        X = MatrixXd::Ones(shape[0], shape[1]);
        for (int i = 0; i < shape[0]; i++) {
            s = PyList_GetItem(p, i);
            X(i, 0) = PyFloat_AsDouble(s);
        }
    }
    else return 0;

    // Parsing Y
    shape[0] = (int) PyList_Size(q);

    VectorXd Y = VectorXd::Ones(shape[0]);
    for (int i = 0; i < shape[0]; i++) {
        s = PyList_GetItem(q, i);
        Y(i) = PyFloat_AsDouble(s);
    }

    // Regression
    Reg List(shape[1] + 1);
    List = one_step_fit(X, Y, z, tol, max_iter);
    VectorXd theta = List.theta;
    int iter = List.iter;
    double tau = List.tau;

    PyObject* result;
    result = PyTuple_New(3);
    PyObject* coef;
    coef = PyList_New(0);
    for (int i = 0; i < theta.size(); i++) {
        PyList_Append(coef, PyFloat_FromDouble(theta(i)));
    }
    PyTuple_SetItem(result, 0, coef);
    PyTuple_SetItem(result, 1, PyFloat_FromDouble(tau));
    PyTuple_SetItem(result, 2, Py_BuildValue("i", iter));

    return result;
}

static PyObject* two_step(PyObject*, PyObject* args, PyObject* kw) {
    double tol = 1e-5, constTau = 1.345;
    int max_iter = 500;
    int shape[2];
    PyObject* p, * q, * r, * s;

    static const char* kwlist[] = { "X", "Y", "tol", "max_iter", NULL };

    if (!PyArg_ParseTupleAndKeywords(args, kw, "OO|ddi", const_cast<char**>(kwlist), &p, &q, &tol, &constTau,&max_iter)) {
        return NULL;
    }

    // Parsing X
    shape[0] = (int) PyList_Size(p);
    r = PyList_GetItem(p, 0);
    MatrixXd X;
    if (PyList_Check(r)) {
        shape[1] = (int)PyList_Size(r);

        X = MatrixXd::Ones(shape[0], shape[1]);
        for (int i = 0; i < shape[0]; i++) {
            for (int j = 0; j < shape[1]; j++) {
                r = PyList_GetItem(p, i);
                s = PyList_GetItem(r, j);
                X(i, j) = PyFloat_AsDouble(s);
            }
        }
    }
    else if (PyFloat_Check(r)) {
        shape[1] = 1;

        X = MatrixXd::Ones(shape[0], shape[1]);
        for (int i = 0; i < shape[0]; i++) {
            s = PyList_GetItem(p, i);
            X(i, 0) = PyFloat_AsDouble(s);
        }
    }
    else return 0;

    // Parsing Y
    shape[0] = (int) PyList_Size(q);

    VectorXd Y = VectorXd::Ones(shape[0]);
    Y.setZero(shape[0]);
    for (int i = 0; i < shape[0]; i++) {
        s = PyList_GetItem(q, i);
        Y(i) = PyFloat_AsDouble(s);
    }

    // Regression
    Reg List(shape[1] + 1);
    List = two_step_fit(X, Y, tol, constTau, max_iter);
    VectorXd theta = List.theta;
    int iter = List.iter;
    double tau = List.tau;

    PyObject* result;
    result = PyTuple_New(3);
    PyObject* coef;
    coef = PyList_New(0);
    for (int i = 0; i < theta.size(); i++) {
        PyList_Append(coef, PyFloat_FromDouble(theta(i)));
    }
    PyTuple_SetItem(result, 0, coef);
    PyTuple_SetItem(result, 1, PyFloat_FromDouble(tau));
    PyTuple_SetItem(result, 2, Py_BuildValue("i", iter));

    return result;
}

static PyMethodDef tfHuber_methods[] = {
    {"mean", (PyCFunction)mean, METH_VARARGS | METH_KEYWORDS, nullptr},
    {"cov", (PyCFunction)cov, METH_VARARGS | METH_KEYWORDS, nullptr},
    {"one_step_reg", (PyCFunction)one_step, METH_VARARGS | METH_KEYWORDS, nullptr},
    {"two_step_reg", (PyCFunction)two_step, METH_VARARGS | METH_KEYWORDS, nullptr},
    {nullptr, nullptr, 0, nullptr}
};

static PyModuleDef tfHuber_module = {
    PyModuleDef_HEAD_INIT,
    "tfHuber",
    "",
    0,
    tfHuber_methods
};

PyMODINIT_FUNC PyInit_tfHuber() {
    return PyModule_Create(&tfHuber_module);
}