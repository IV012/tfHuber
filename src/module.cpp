#include <Eigen/Dense>
#include <Python.h>
#include <numpy/arrayobject.h>
#include <string.h>
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

double f4(const double x, const MatrixXd Y, const VectorXd sqnorm, const int n, const int d) {
    int N = n * (n - 1) >> 1;
    MatrixXd result = MatrixXd::Zero(d, d);
    for (int i = 0; i < N; i++) {
        if (sqnorm(i) * sqnorm(i) / 4 >= x) {
            result = result + Y.row(i).transpose() * Y.row(i) / sqnorm(i) * x;
        }
        else {
            result = result + Y.row(i).transpose() * Y.row(i) * sqnorm(i) / 4;
        }
    }
    result = result / x / N;
    return result.norm() - (log(2 * d) + log(n)) / n;
}

double rootf4(const MatrixXd Y, const VectorXd sqnorm, const int n, const int d, double low, double up, const double tol = 1e-5, const int max_iter = 500) {
    int iter = 0;
    double mid, val;
    while (iter <= max_iter && (up - low) > tol) {
        mid = (up + low) / 2;
        val = f4(mid, Y, sqnorm, n, d);
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

typedef struct Mean {
    int iter;
    double mu;
    double tau;
    Mean() : iter(0), mu(0), tau(0) {}
}Mean;

int sgn(const double x) {
    return (x > 0) - (x < 0);
}

double HuberDer(const VectorXd res, const double tau, const int n) {
    double rst = 0;
    for (int i = 0; i < n; i++) {
        double cur = res(i);
        rst -= abs(cur) <= tau ? cur : tau * sgn(cur);
    }
    return rst / n;
}

Mean Huber_mean(VectorXd x, const int grad = true, const double tol = 1e-5, const int max_iter = 500) {
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

    if (grad == false) {
        double muold = 0;
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
    }
    else if (grad) {
        double mx = x.mean();
        x = (x.array() - mx).matrix();

        double derold = HuberDer(x, taunew, n);
        double mudiff = -derold;
        munew = derold;

        VectorXd res = (x.array() - munew).matrix();
        sqres = res.array().square().matrix();
        taunew = sqrt((long double)rootf1(sqres, n, sqres.minCoeff(), sqres.sum()));
        double dernew = HuberDer(res, taunew, n);
        double derdiff = dernew - derold;
        tauold = taunew;

        iter++;
        while (abs(dernew) > tol && iter <= max_iter) {
            double alpha = 1;
            double cross = mudiff * derdiff;
            if (cross > 0) {
                double a1 = cross / derdiff * derdiff;
                double a2 = mudiff * mudiff / cross;
                alpha = min(min(a1, a2), 100.0);
            }
            derold = dernew;
            mudiff = -alpha * dernew;
            munew += mudiff;

            res = (x.array() - munew).matrix();
            sqres = res.array().square().matrix();
            tauold = taunew;
            taunew = sqrt((long double)rootf1(sqres, n, sqres.minCoeff(), sqres.sum()));
            dernew = HuberDer(res, taunew, n);
            derdiff = dernew - derold;

            iter++;
        }
        munew += mx;
        taunew = tauold;
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

MatrixXd Huber_cov(const MatrixXd X, const char* type = "element", const int pairwise = false, const double tol = 1e-5, const int max_iter = 500) {
    int n = (int)X.rows();
    int p = (int)X.cols();
    VectorXd mu(p);
    MatrixXd sigma = MatrixXd::Zero(p, p);

    if (strcmp(type, "element") == 0) {
        for (int j = 0; j < p; j++) {
            mu(j) = Huber_mean(X.col(j), true, tol, max_iter).mu;
            double theta = Huber_mean(X.col(j).array().square().matrix(), true, tol, max_iter).mu;
            double temp = mu(j) * mu(j);
            if (theta > temp) {
                theta -= temp;
            }
            sigma(j, j) = theta;
        }
        if (pairwise) {
            for (int i = 0; i < p - 1; i++) {
                for (int j = i + 1; j < p; j++) {
                    sigma(i, j) = sigma(j, i) = Huber_mean((X.col(i).array() - mu(i)).cwiseProduct(X.col(j).array() - mu(j)).matrix()).mu;
                }
            }
        }
        else {
            int N = n * (n - 1) >> 1;
            MatrixXd Y(N, p);
            for (int i = 0, k = 0; i < n - 1; i++) {
                for (int j = i + 1; j < n; j++) {
                    Y.row(k++) = X.row(i) - X.row(j);
                }
            }
            for (int i = 0; i < p - 1; i++) {
                for (int j = i + 1; j < p; j++) {
                    sigma(i, j) = sigma(j, i) = hMeanCov(Y.col(i).cwiseProduct(Y.col(j)) / 2, n, p, N, tol, max_iter);
                }
            }
        }
    }
    else if (strcmp(type, "spectrum") == 0){
        int N = n * (n - 1) >> 1;
        MatrixXd Y(N, p);
        for (int i = 0, k = 0; i < n - 1; i++) {
            for (int j = i + 1; j < n; j++) {
                Y.row(k++) = X.row(i) - X.row(j);
            }
        }
        VectorXd sqnorm(N);
        for (int i = 0; i < N; i++) {
            sqnorm(i) = Y.row(i).squaredNorm();
        }
        double tau = sqrt((long double)rootf4(Y, sqnorm, n, p, sqnorm.minCoeff(), sqnorm.sum()));
        for (int i = 0; i < N; i++) {
            sigma = sigma + min(tau, sqnorm(i) / 2) * Y.row(i).transpose() * Y.row(i) / sqnorm(i);
        }
        sigma = sigma / N;
    }
    return sigma;
}


typedef struct Reg {
    int iter;
    VectorXd theta;
    double tau;

    Reg(int n) {
        iter = 0;
        tau = 0;
        theta = VectorXd::Zero(n);
    }
}Reg;

void updateHuber(const MatrixXd Z, const VectorXd res, VectorXd& der, VectorXd& grad, const int n, const double tau) {
    for (int i = 0; i < n; i++) {
        double cur = res(i);
        if (abs(cur) <= tau) {
            der(i) = -cur;
        }
        else {
            der(i) = -tau * sgn(cur);
        }
    }
    grad = Z.transpose() * der;
    grad = grad / n;
}

MatrixXd standardize_data(MatrixXd X, VectorXd m, VectorXd sx, int d) {
    for (int i = 0; i < d; i++) {
        X.col(i) = (X.col(i).array() - m(i)).matrix();
        X.col(i) = X.col(i) / sx(i);
    }
    return X;
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

double mad(const VectorXd res) {
    double mid = getMidValue(res);
    double mad = getMidValue((res.array() - mid).cwiseAbs().matrix()) / 0.6744898;
    return mad;
}

Reg one_step_fit(const MatrixXd X, VectorXd Y, const int grad = true, double z = 0, const double tol = 1e-5, const int max_iter = 500) {
    int n = (int)X.rows();
    int d = (int)X.cols();
    MatrixXd Z(n, d + 1);
    Z.block(0, 0, n, 1) = VectorXd::Ones(n);
    VectorXd sx = VectorXd::Zero(d);
    VectorXd m = VectorXd::Zero(d);
    double my;

    for (int i = 0; i < d; i++) {
        m(i) = X.col(i).mean();
        for (int j = 0; j < n; j++) {
            sx(i) += (X(j, i) - m(i)) * (X(j, i) - m(i));
        }
    }
    sx = (sx / (n - 1)).array().sqrt().matrix();
    Z.block(0, 1, n, d) = standardize_data(X, m, sx, d);
    my = Y.mean();
    Y = (Y.array() - my).matrix();
    if (z == 0) {
        z = log(n);
    }

    VectorXd thetaold = VectorXd::Zero(d + 1);
    VectorXd thetanew(d + 1);

    double sigma;
    VectorXd res(n);
    VectorXd sqres(n);
    sqres = Y - Z * thetanew;
    for (int i = 0; i < n; i++) {
        sqres(i) *= sqres(i);
    }
    sigma = sqrt(sqres.sum() / n);

    double tauold = 0;
    double taunew = sigma * sqrt((d + z) / n);
    int iter = 0;

    if (grad == false) {
        MatrixXd WZ(n, d + 1);
        VectorXd WY(n);
        double w;
        thetanew = (Z.transpose() * Z).inverse() * (Z.transpose() * Y);

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
    }
    else if (grad) {
        VectorXd der(n);
        VectorXd gradold(d + 1), gradnew(d + 1);
        res = Y;
        taunew = 1.345 * mad(res);
        updateHuber(Z, res, der, gradold, n, taunew);
        thetanew = -gradold;
        VectorXd thetadiff = -gradold;

        res = Y - Z * thetanew;
        sqres = res.array().square().matrix();
        taunew = sqrt(rootf2(sqres, n, d, z, sqres.minCoeff(), sqres.sum()));
        updateHuber(Z, res, der, gradnew, n, taunew);
        VectorXd graddiff = gradnew - gradold;
        iter++;

        while (gradnew.lpNorm<Infinity>() > tol && iter < max_iter) {
            double alpha = 1;
            double cross = thetadiff.dot(graddiff);
            if (cross > 0) {
                double a1 = cross / (graddiff.squaredNorm());
                double a2 = (thetadiff.squaredNorm()) / cross;
                alpha = min(min(a1, a2), 100.0);
            }
            gradold = gradnew;
            thetadiff = -alpha * gradnew;
            thetanew += thetadiff;
            res -= Z * thetadiff;
            tauold = taunew;
            taunew = sqrt(rootf2(sqres, n, d, z, sqres.minCoeff(), sqres.sum()));
            updateHuber(Z, res, der, gradnew, n, taunew);
            graddiff = gradnew - gradold;
            iter++;
        }
        taunew = tauold;
    }

    for (int i = 1; i <= d; i++) {
        thetanew(i) = thetanew(i) / sx(i - 1);
    }
    Y = (Y.array() + my).matrix();
    thetanew(0) = Huber_mean(Y - X * thetanew.block(1, 0, d, 1), grad).mu;

    Reg List(d + 1);
    List.iter = iter;
    List.tau = taunew;
    List.theta = thetanew;
    return List;
}

Reg two_step_fit(const MatrixXd X, VectorXd Y, const int grad = true, const double tol = 1e-5, const double constTau = 1.345, const int max_iter = 500) {
    int n = (int)X.rows();
    int d = (int)X.cols();
    MatrixXd Z(n, d + 1);
    Z.block(0, 0, n, 1) = VectorXd::Ones(n);
    VectorXd sx = VectorXd::Zero(d);
    VectorXd m = VectorXd::Zero(d);
    double my;
    for (int i = 0; i < d; i++) {
        m(i) = X.col(i).mean();
        for (int j = 0; j < n; j++) {
            sx(i) += (X(j, i) - m(i)) * (X(j, i) - m(i));
        }
    }
    sx = (sx / (n - 1)).array().sqrt().matrix();
    Z.block(0, 1, n, d) = standardize_data(X, m, sx, d);
    my = Y.mean();
    Y = (Y.array() - my).matrix();


    VectorXd thetaold = VectorXd::Zero(d + 1);
    VectorXd thetanew(d + 1);

    double sigma;
    VectorXd res(n);
    VectorXd sqres(n);
    sqres = Y - Z * thetanew;
    for (int i = 0; i < n; i++) {
        sqres(i) *= sqres(i);
    }
    sigma = sqrt((long double)sqres.sum() / ((long double)n - (long double)d));

    double tauold = 0;
    double taunew = sigma * sqrt((long double)n / log((long double)(d + log(n * d))));
    int iter = 0;

    if (grad == false) {
        MatrixXd WZ(n, d + 1);
        VectorXd WY(n);
        double w;

        thetanew = (Z.transpose() * Z).inverse() * (Z.transpose() * Y);

        while ((abs(taunew - tauold) > tol || (thetanew - thetaold).lpNorm<Infinity>() > tol) && iter < max_iter) {
            thetaold = thetanew;
            tauold = taunew;
            res = Y - Z * thetaold;
            taunew = constTau * mad(res);
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
    }
    else if (grad) {
        VectorXd der(n);
        VectorXd gradold(d + 1), gradnew(d + 1);
        res = Y;
        taunew = constTau * mad(res);
        updateHuber(Z, res, der, gradold, n, taunew);
        thetanew = -gradold;
        VectorXd thetadiff = -gradold;

        res = Y - Z * thetanew;
        taunew = constTau * mad(res);
        updateHuber(Z, res, der, gradnew, n, taunew);
        VectorXd graddiff = gradnew - gradold;
        iter++;

        while (gradnew.lpNorm<Infinity>() > tol && iter < max_iter) {
            double alpha = 1;
            double cross = thetadiff.dot(graddiff);
            if (cross > 0) {
                double a1 = cross / (graddiff.squaredNorm());
                double a2 = (thetadiff.squaredNorm()) / cross;
                alpha = min(min(a1, a2), 100.0);
            }
            gradold = gradnew;
            thetadiff = -alpha * gradnew;
            thetanew += thetadiff;
            res -= Z * thetadiff;
            tauold = taunew;
            taunew = constTau * mad(res);
            updateHuber(Z, res, der, gradnew, n, taunew);
            graddiff = gradnew - gradold;
            iter++;
        }
        taunew = tauold;
    }


    for (int i = 1; i <= d; i++) {
        thetanew(i) = thetanew(i) / sx(i - 1);
    }
    Y = (Y.array() + my).matrix();
    thetanew(0) = Huber_mean(Y - X * thetanew.block(1, 0, d, 1), grad).mu;
    Reg List(d + 1);
    List.iter = iter;
    List.tau = taunew;
    List.theta = thetanew;
    return List;
}

VectorXd softThresh(const VectorXd x, const VectorXd lambda) {
    VectorXd res(x.size());
    res = x.cwiseAbs() - lambda;
    for (int i = 0; i < x.size(); i++) {
        if (res(i) < 0) {
            res(i) = 0;
        }
    }
    return x.cwiseSign().cwiseProduct(res);
}

VectorXd cmptLambda(const VectorXd beta, const double lambda) {
    VectorXd rst = lambda * VectorXd::Ones(beta.size());
    rst(0) = 0;
    return rst;
}

double loss(const VectorXd Y, const VectorXd Ynew, const int lossType, const double tau) {
    double rst = 0;
    if (lossType == 0) {
        rst = (Y - Ynew).array().square().mean() / 2;
    }
    else if (lossType == 1) {
        VectorXd res = Y - Ynew;
        for (int i = 0; i < Y.size(); i++) {
            if (abs(res(i)) <= tau) {
                rst += res(i) * res(i) / 2;
            }
            else {
                rst += tau * abs(res(i)) - tau * tau / 2;
            }
        }
        rst /= double(Y.size());
    }
    return rst;
}

VectorXd gradLoss(const MatrixXd X, const VectorXd Y, const VectorXd beta, const int lossType, const double tau) {
    VectorXd res = Y - X * beta;
    VectorXd rst = VectorXd::Zero(beta.size());
    if (lossType == 0) {
        rst = -1 * (res.transpose() * X).transpose();
    }
    else if (lossType == 1) {
        for (int i = 0; i < Y.size(); i++) {
            if (abs(res(i)) <= tau) {
                rst -= res(i) * X.row(i).transpose();
            }
            else {
                rst -= tau * sgn(res(i)) * X.row(i).transpose();
            }
        }
    }
    return rst / double(Y.size());
}

VectorXd updateBeta(const MatrixXd X, const VectorXd Y, VectorXd beta, const double phi, const VectorXd Lambda, const int lossType, const double tau) {
    VectorXd first = beta - gradLoss(X, Y, beta, lossType, tau) / phi;
    VectorXd second = Lambda / phi;
    return softThresh(first, second);
}

double cmptPsi(const MatrixXd X, const VectorXd Y, const VectorXd betaNew, const VectorXd beta, const double phi, const int lossType, const double tau) {
    VectorXd diff = betaNew - beta;
    double rst = loss(Y, X * beta, lossType, tau) + double(gradLoss(X, Y, beta, lossType, tau).transpose() * diff) + double(diff.transpose() * diff) * phi / 2;
    return rst;
}

typedef struct Beta {
    VectorXd beta;
    double phi;

    Beta(int n) {
        phi = 0;
        beta = VectorXd::Zero(n);
    }
}Beta;

Beta LAMM(const MatrixXd X, const VectorXd Y, const VectorXd Lambda, VectorXd beta, const double phi, const int lossType, const double tau, const double gamma) {
    double phiNew = phi;
    VectorXd betaNew;
    double FVal, PsiVal;
    while (true) {
        betaNew = updateBeta(X, Y, beta, phiNew, Lambda, lossType, tau);
        FVal = loss(Y, X * betaNew, lossType, tau);
        PsiVal = cmptPsi(X, Y, betaNew, beta, phiNew, lossType, tau);
        if (FVal <= PsiVal) {
            break;
        }
        phiNew *= gamma;
    }
    Beta result((int)betaNew.size());
    result.beta = betaNew;
    result.phi = phiNew;
    return result;
}

VectorXd lasso(const MatrixXd X, const VectorXd Y, const double lambda, const double phi0 = 0.001, const double gamma = 1.5, const double epsilon_c = 0.001, const int max_iter = 500) {
    int d = (int)X.cols() - 1;
    VectorXd beta = VectorXd::Zero(d + 1);
    VectorXd betaNew = VectorXd::Zero(d + 1);
    VectorXd Lambda = cmptLambda(beta, lambda);
    double phi = phi0;
    int iter = 0;
    Beta listLAMM(d + 1);
    while (iter < max_iter) {
        iter++;
        listLAMM = LAMM(X, Y, Lambda, beta, phi, 0, 1.0, gamma);
        betaNew = listLAMM.beta;
        phi = listLAMM.phi;
        phi = max(phi0, phi / gamma);
        if ((betaNew - beta).lpNorm<Infinity>() <= epsilon_c) {
            break;
        }
        beta = betaNew;
    }
    return betaNew;
}

Reg huberLasso(const MatrixXd X, const VectorXd Y, const double lambda, double tau = -1, const double constTau = 1.345, const double phi0 = 0.001, const double gamma = 1.5, const double epsilon_c = 0.001, const int max_iter = 500) {
    int n = (int)X.rows();
    int d = (int)X.cols() - 1;
    VectorXd beta = VectorXd::Zero(d + 1);
    VectorXd betaNew = VectorXd::Zero(d + 1);
    VectorXd Lambda = cmptLambda(beta, lambda);
    if (tau < 0) {
        VectorXd betaLasso = lasso(X, Y, lambda, phi0, gamma, epsilon_c, max_iter);
        VectorXd res = Y - X * betaLasso;
        tau = constTau * mad(res);
    }
    double phi = phi0;
    int iter = 0;
    Beta listLAMM(d + 1);
    VectorXd res(n);
    while (iter < max_iter) {
        iter++;
        listLAMM = LAMM(X, Y, Lambda, beta, phi, 1, tau, gamma);
        betaNew = listLAMM.beta;
        phi = listLAMM.phi;
        phi = max(phi0, phi / gamma);
        if ((betaNew - beta).lpNorm<Infinity>() <= epsilon_c) {
            break;
        }
        beta = betaNew;
        res = Y - X * beta;
        tau = constTau * mad(res);
    }
    Reg List(d + 1);
    List.iter = iter;
    List.tau = tau;
    List.theta = betaNew;
    return List;
}

ArrayXi getIndex(const int n, const int low, const int up) {
    VectorXi index(n);
    int k = 0;
    for (int i = 0; i < n; i++) {
        if (i >= low && i <= up) {
            index(k) = i;
            k++;
        }
    }
    return index.block(0, 0, k, 1).array();
}

ArrayXi getIndexComp(const int n, const int low, const int up) {
    VectorXi index(n);
    int k = 0;
    for (int i = 0; i < n; i++) {
        if (i < low || i > up) {
            index(k) = i;
            k++;
        }
    }
    return index.block(0, 0, k, 1).array();
}

double pairPred(const MatrixXd X, const VectorXd Y, const VectorXd beta) {
    int n = (int)X.rows();
    int d = (int)X.cols() - 1;
    int m = n * (n - 1) >> 1;
    MatrixXd pairX(m, d + 1);
    VectorXd pairY(m);
    int k = 0;
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            pairX.row(k) = X.row(i) - X.row(j);
            pairY(k++) = Y(i) - Y(j);
        }
    }
    VectorXd predY = pairX * beta;
    return (pairY - predY).array().square().sum();
}

typedef struct lassoReg {
    int iter;
    VectorXd theta;
    double tau;
    double lambdaMin;

    lassoReg(int n) {
        iter = 0;
        tau = 0;
        lambdaMin = 0;
        theta = VectorXd::Zero(n);
    }
}lassoReg;

lassoReg cvHuberLasso(const MatrixXd X, const VectorXd Y, VectorXd lSeq = VectorXd::Zero(1), int nlambda = 30, const double constTau = 2.5, const double phi0 = 0.001, const double gamma = 1.5, const double epsilon_c = 0.001, const int max_iter = 500, int nfolds = 3) {
    int n = (int)X.rows();
    int d = (int)X.cols();
    MatrixXd Z(n, d + 1);
    Z.block(0, 1, n, d) = X;
    Z.block(0, 0, n, 1) = VectorXd::Ones(n);
    VectorXd lambdaSeq(nlambda);
    if (lSeq.isZero()) {
        double lambdaMax = log(((long double)(Y.transpose() * Z).cwiseAbs().maxCoeff() / n));
        double lambdaMin = log(0.01) + lambdaMax;
        lambdaSeq = VectorXd::LinSpaced(nlambda, lambdaMin, lambdaMax);
        lambdaSeq = lambdaSeq.array().exp().matrix();
    }
    else {
        lambdaSeq = lSeq;
        nlambda = (int)lambdaSeq.size();
    }
    if (nfolds > 10 || nfolds > n) {
        nfolds = n < 10 ? n : 10;
    }
    int size = n / nfolds;
    VectorXd mse = VectorXd::Zero(nlambda);
    int low, up;
    ArrayXi idx, idxComp;
    VectorXd thetaHat(d + 1);
    Reg hLassoList(d + 1);
    for (int i = 0; i < nlambda; i++) {
        for (int j = 0; j < nfolds; j++) {
            low = j * size;
            up = (j == (nfolds - 1)) ? (n - 1) : ((j + 1) * size - 1);
            idx = getIndex(n, low, up);
            idxComp = getIndexComp(n, low, up);
            hLassoList = huberLasso(Z(idxComp, all), Y(idxComp), lambdaSeq(i), -1, constTau, phi0, gamma, epsilon_c, max_iter);
            thetaHat = hLassoList.theta;
            mse(i) += pairPred(Z(idx, all), Y(idx, all), thetaHat);
        }
    }
    int cvIdx;
    mse.minCoeff(&cvIdx);
    hLassoList = huberLasso(Z, Y, lambdaSeq(cvIdx), -1, constTau, phi0, gamma, epsilon_c, max_iter);
    VectorXd theta = hLassoList.theta;
    Mean listMean = Huber_mean(Y - X * theta.block(1, 0, d, 1));
    theta(0) = listMean.mu;
    lassoReg List(d + 1);
    List.lambdaMin = lambdaSeq(cvIdx);
    List.theta = theta;
    List.tau = hLassoList.tau;
    List.iter = hLassoList.iter;

    return List;
}
static PyObject* mean(PyObject*, PyObject* args, PyObject* kw) {
    double tol = 1e-5;
    int max_iter = 500;
    int grad = true;
    int n;
    PyArrayObject* data;
    static const char* kwlist[] = { "X", "grad", "tol", "max_iter", NULL };

    if (!PyArg_ParseTupleAndKeywords(args, kw, "O|idi", const_cast<char**>(kwlist), &data, &grad, &tol, &max_iter)) {
        return NULL;
    }
    /* Check the dimension of the array */
    if (data->nd != 1)
    {
        PyErr_SetString(PyExc_TypeError, "Expected a 1-dimensional array");
        return NULL;
    }

    /* Check the type of items in the array */
    if (data->descr->type_num != PyArray_DOUBLE)
    {
        PyErr_SetString(PyExc_TypeError, "Expected an array of doubles");
        return NULL;
    }

    // Parsing Vector
    n = (int) data->dimensions[0];

    VectorXd X = VectorXd::Zero(n);
    for (int i = 0; i < data->dimensions[0]; i++) {
        X(i) = *(double*)PyArray_GETPTR1(data, i);
    }

    // Huber Mean
    Mean List;
    List = Huber_mean(X, grad, tol, max_iter);
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
    const char* type = "element";
    int pairwise = false;
    int max_iter = 500;
    PyArrayObject* data;

    static const char* kwlist[] = {"X", "type", "pairwise", "tol", "max_iter", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kw, "O|sidi", const_cast<char**>(kwlist), &data, &type, &pairwise, &tol, &max_iter)) {
        return NULL;
    }

    // Parsing Matrix

    if (data->nd != 2)
    {
        PyErr_SetString(PyExc_TypeError, "Expected a 2-dimensional array");
        return NULL;
    }

    /* Check the type of items in the array */
    if (data->descr->type_num != PyArray_DOUBLE)
    {
        PyErr_SetString(PyExc_TypeError, "Expected an array of doubles");
        return NULL;
    }
    MatrixXd X = MatrixXd::Zero(data->dimensions[0], data->dimensions[1]);
    for (int i = 0; i < data->dimensions[0]; i++) {
        for (int j = 0; j < data->dimensions[1]; j++) {
            X(i, j) = *(double*)PyArray_GETPTR2(data, i, j);
        }
    }
        
    // Huber Covariance
    MatrixXd COV;
    COV = Huber_cov(X, type, pairwise, tol, max_iter);


    PyObject* result;
    result = PyList_New(0);
    for (int i = 0; i < (int)COV.rows(); i++) {
        PyObject* coef;
        coef = PyList_New(0);
        for (int j = 0; j < (int)COV.cols(); j++) {
            PyList_Append(coef, PyFloat_FromDouble(COV(i, j)));
        }
        PyList_Append(result, coef);
    }
    return result;
}

static PyObject* one_step(PyObject*, PyObject* args, PyObject* kw) {
    double z = 0, tol = 1e-5;
    int grad = true;
    int max_iter = 500;
    PyArrayObject* p, * q;

    static const char* kwlist[] = { "X", "Y", "grad", "tol", "max_iter", NULL };

    if (!PyArg_ParseTupleAndKeywords(args, kw, "OO|iddi", const_cast<char**>(kwlist), &p, &q, &grad, &tol, &max_iter)) {
        return NULL;
    }

    // Parsing X
    if (p->nd != 2)
    {
        PyErr_SetString(PyExc_TypeError, "Expected a 2-dimensional array");
        return NULL;
    }
    if (p->descr->type_num != PyArray_DOUBLE)
    {
        PyErr_SetString(PyExc_TypeError, "Expected an array of doubles");
        return NULL;
    }
    MatrixXd X = MatrixXd::Zero(p->dimensions[0], p->dimensions[1]);
    for (int i = 0; i < p->dimensions[0]; i++) {
        for (int j = 0; j < p->dimensions[1]; j++) {
            X(i, j) = *(double*)PyArray_GETPTR2(p, i, j);
        }
    }

    // Parsing Y
    if (q->nd != 1)
    {
        PyErr_SetString(PyExc_TypeError, "Expected a 1-dimensional array");
        return NULL;
    }
    if (q->descr->type_num != PyArray_DOUBLE)
    {
        PyErr_SetString(PyExc_TypeError, "Expected an array of doubles");
        return NULL;
    }
    int n = (int)q->dimensions[0];

    VectorXd Y = VectorXd::Zero(n);
    for (int i = 0; i < q->dimensions[0]; i++) {
        Y(i) = *(double*)PyArray_GETPTR1(q, i);
    }

    // Regression
    Reg List(1);
    List = one_step_fit(X, Y, grad, z, tol, max_iter);
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
    int grad = true;
    int max_iter = 500;
    PyArrayObject* p, * q;

    static const char* kwlist[] = {"X", "Y", "grad", "tol", "constTau", "max_iter", NULL };

    if (!PyArg_ParseTupleAndKeywords(args, kw, "OO|iddi", const_cast<char**>(kwlist), &p, &q, &grad, &tol, &constTau, &max_iter)) {
        return NULL;
    }

    // Parsing X
    if (p->nd != 2)
    {
        PyErr_SetString(PyExc_TypeError, "Expected a 2-dimensional array");
        return NULL;
    }
    if (p->descr->type_num != PyArray_DOUBLE)
    {
        PyErr_SetString(PyExc_TypeError, "Expected an array of doubles");
        return NULL;
    }
    MatrixXd X = MatrixXd::Zero(p->dimensions[0], p->dimensions[1]);
    for (int i = 0; i < p->dimensions[0]; i++) {
        for (int j = 0; j < p->dimensions[1]; j++) {
            X(i, j) = *(double*)PyArray_GETPTR2(p, i, j);
        }
    }

    // Parsing Y
    if (q->nd != 1)
    {
        PyErr_SetString(PyExc_TypeError, "Expected a 1-dimensional array");
        return NULL;
    }
    if (q->descr->type_num != PyArray_DOUBLE)
    {
        PyErr_SetString(PyExc_TypeError, "Expected an array of doubles");
        return NULL;
    }
    int n = (int)q->dimensions[0];

    VectorXd Y = VectorXd::Zero(n);
    for (int i = 0; i < q->dimensions[0]; i++) {
        Y(i) = *(double*)PyArray_GETPTR1(q, i);
    }

    // Regression
    Reg List(1);
    List = two_step_fit(X, Y, grad, tol, constTau, max_iter);
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

static PyObject* cvlasso(PyObject*, PyObject* args, PyObject* kw) {
    double constTau = 2.5, phi0 = 0.001, gamma = 1.5, epsilon_c = 0.001;
    int nlambda = 30, nfolds = 3;
    int max_iter = 500;
    PyArrayObject* p, * q;
    PyObject* l = Py_BuildValue("i", 1), * s;

    static const char* kwlist[] = { "X", "Y", "lSeq", "nlambda", "constTau", "phi0", "gamma", "tol", "max_iter", "nfolds", NULL };

    if (!PyArg_ParseTupleAndKeywords(args, kw, "OO|Oiddddii", const_cast<char**>(kwlist), &p, &q, &l, &nlambda, &constTau, &phi0, &gamma, &epsilon_c, &max_iter, &nfolds)) {
        return NULL;
    }

    // Parsing X
    if (p->nd != 2)
    {
        PyErr_SetString(PyExc_TypeError, "Expected a 2-dimensional array");
        return NULL;
    }
    if (p->descr->type_num != PyArray_DOUBLE)
    {
        PyErr_SetString(PyExc_TypeError, "Expected an array of doubles");
        return NULL;
    }
    MatrixXd X = MatrixXd::Zero(p->dimensions[0], p->dimensions[1]);
    for (int i = 0; i < p->dimensions[0]; i++) {
        for (int j = 0; j < p->dimensions[1]; j++) {
            X(i, j) = *(double*)PyArray_GETPTR2(p, i, j);
        }
    }

    // Parsing Y
    if (q->nd != 1)
    {
        PyErr_SetString(PyExc_TypeError, "Expected a 1-dimensional array");
        return NULL;
    }
    if (q->descr->type_num != PyArray_DOUBLE)
    {
        PyErr_SetString(PyExc_TypeError, "Expected an array of doubles");
        return NULL;
    }
    int n = (int)q->dimensions[0];

    VectorXd Y = VectorXd::Zero(n);
    for (int i = 0; i < q->dimensions[0]; i++) {
        Y(i) = *(double*)PyArray_GETPTR1(q, i);
    }

    // Parsing lambda sequence
    VectorXd lSeq = VectorXd::Zero(1);
    if (PyList_Check(l)) {
        int l_size = (int)PyList_Size(l);
        lSeq = VectorXd::Zero(l_size);
        for (int i = 0; i < l_size; i++) {
            s = PyList_GetItem(l, i);
            lSeq(i) = PyFloat_AsDouble(s);
        }
    }
    // Regression
    lassoReg List(1);
    List = cvHuberLasso(X, Y, lSeq, nlambda, constTau, phi0, gamma, epsilon_c, max_iter, nfolds);
    VectorXd theta = List.theta;
    int iter = List.iter;
    double tau = List.tau;
    double lambda_min = List.lambdaMin;

    PyObject* result;
    result = PyTuple_New(4);
    PyObject* coef;
    coef = PyList_New(0);
    for (int i = 0; i < theta.size(); i++) {
        PyList_Append(coef, PyFloat_FromDouble(theta(i)));
    }
    PyTuple_SetItem(result, 0, coef);
    PyTuple_SetItem(result, 1, PyFloat_FromDouble(tau));
    PyTuple_SetItem(result, 2, Py_BuildValue("i", iter));
    PyTuple_SetItem(result, 3, PyFloat_FromDouble(lambda_min));

    return result;
}

static PyMethodDef tfHuber_methods[] = {
    {"mean", (PyCFunction)mean, METH_VARARGS | METH_KEYWORDS, "Adaptive Huber mean estimation."},
    {"cov", (PyCFunction)cov, METH_VARARGS | METH_KEYWORDS, "Adaptive Huber covariance matrix estimation."},
    {"one_step_reg", (PyCFunction)one_step, METH_VARARGS | METH_KEYWORDS, "One-step adaptive Huber regression."},
    {"two_step_reg", (PyCFunction)two_step, METH_VARARGS | METH_KEYWORDS, "Two-step adaptive Huber regression."},
    {"cvlasso", (PyCFunction)cvlasso, METH_VARARGS | METH_KEYWORDS, "K-fold cross validated Huber-Lasso regression."},
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