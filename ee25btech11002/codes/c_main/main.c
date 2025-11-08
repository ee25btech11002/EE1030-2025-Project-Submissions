#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// Helper Functions
// r = rows;     
// c = columns;


//formation of matrix A
// using calloc instead of malloc because ; it automatically sets the numbers in the matrix to 0
double **matrix(int r,int c)
{
    double **A = malloc(r * sizeof(double *));
    for (int i = 0; i < r; i++)
    {    A[i] = calloc(c, sizeof(double)); }
    return A;
}

//function to free the memory of matrix after I am done with the program
void free_matrix(double **A,int r)
{
    for (int i = 0;i<r; i++)
    {    free(A[i]);  }
    free(A);
}

//function to calculate the transpose of the matrix
void transpose(double **A,double **AT,int m,int n)
{
    for (int i=0;i<m;i++)
        for (int j=0;j < n;j++)
            AT[j][i] = A[i][j];
}

//function for multiplication of matrices
//avoided to initialize C matrix to zero
void matrix_multiply(double **A,double **B,double **C,int m,int n,int p)
{
    for(int i=0;i<m;i++)
    {
        for(int j=0;j<n;j++)
        {
            double sum = 0;
            for(int k = 0;k < p;k++)
            {
                sum += A[i][k] * B[k][j];
            }
            C[i][j] = sum;
        }
    }
}

//function to orthonormalize matrix X ; required in Power Block Iteration
void orthonormalize(double **X,int n,int k)
{
    for(int j=0;j<k;j++)
    {
        double norm = 0;
        for(int i = 0; i < n; i++)
        {
            norm += X[i][j] * X[i][j];
        }
        norm = sqrt(norm);
        if(norm < 1e-12)
        {
            continue;
        }
        for(int i=0;i<n;i++)
        {
            X[i][j] /= norm;
        }
        for (int z = j + 1; z < k; z++)
        {
            double sum = 0;
            for(int i=0;i<n;i++)
            {
                sum += X[i][j] * X[i][z];
            }
            
            for (int i = 0; i < n; i++)
            {
                X[i][z] -= sum * X[i][j];
            }
        }
    }
}   
// this fuction is to read the image in jpg or png or pgm format
//processing it and tmaking the matrix out of it 
double **read_image(const char *filename, int *r, int *c)
{
    int channels;
    unsigned char *img = stbi_load(filename, c, r, &channels, 1); // force grayscale
    

    double **A = matrix(*r, *c);
    for (int i = 0; i < *r; i++)
        for (int j = 0; j < *c; j++)
            A[i][j] = img[i * (*c) + j];

    stbi_image_free(img);
    return A;
}

//this function is to give the output as .png file which will be the compressed size image
void write_image(const char *filename, double **A, int r, int c)
{
    unsigned char *img = malloc(r * c);
    for (int i = 0; i < r; i++)
        for (int j = 0; j < c; j++) {
            int val = (int)fmin(fmax(A[i][j], 0), 255);
            img[i * c + j] = val;
        }
    stbi_write_png(filename, c, r, 1, img, c);
    free(img);
}
double frobenius_norm(double **A, int r, int c)
{
    double sum = 0.0;
    for (int i = 0; i < r; i++)
        for (int j = 0; j < c; j++)
            sum += A[i][j] * A[i][j];
    return sqrt(sum);
}

int main() {
    int m,n,k;
    int max_iter = 10;

    //Reading image
    char input_image[100], output_image[100];
    printf("Enter input image file (PGM/PNG/JPG): ");
    scanf("%s", input_image);

    double **A = read_image(input_image, &m, &n);
    if (!A) {
        printf("Error reading image file.\n");
        return 1;
    }

    printf("Enter k: ");
    scanf("%d", &k);


    // allocating matrices
    double **AT = matrix(n, m);
    double **Y = matrix(m, k);
    double **Z = matrix(n, k);
    double **X = matrix(n, k);

    
    /*
    The process is as follows;
    Initialize X_0 as identity (n×k)
    (AT @ A) @ X_0 = X_1;
    
    where @ indicates multiplication of matrices
    now we have to orthonormalize X_1 and then repeat the same step
    let us suppose X_n is the final matrix;
    
    when the forbenius norm of X_n and X_(n-1) is less than the tolerance(a particular value)
    we have to stop 
    
    thereby the matrix X will contain all the eigen vectors of AT @ A
    */
    
    // Initialize X as identity (n×k)
    for(int i=0;i<n;i++)
        for (int j=0;j<k;j++)
        {
            if(i == j)
            { X[i][j] = 1; }
            else{ X[i][j] = 0; }
        }

    //calculating AT
    transpose(A, AT, m, n);

    //POWER BLOCK ITERATION WITH TOLERANCE
    double tol = 1e-6;
    int iter = 0;
    double diff = 1.0;
    int max_limit = 500; // safety cap

    double **X_prev = matrix(n, k);

    while (diff > tol && iter < max_limit) {
        for (int i = 0; i < n; i++)
            for (int j = 0; j < k; j++)
                X_prev[i][j] = X[i][j];

        matrix_multiply(A, X, Y, m, k, n);
        matrix_multiply(AT, Y, Z, n, k, m);
        orthonormalize(Z, n, k);

        for (int i = 0; i < n; i++)
            for (int j = 0; j < k; j++)
                X[i][j] = Z[i][j];

        diff = 0;
        for (int i = 0; i < n; i++)
            for (int j = 0; j < k; j++)
                diff += (X[i][j] - X_prev[i][j]) * (X[i][j] - X_prev[i][j]);
        diff = sqrt(diff);
        iter++;
    }

    printf("Power iteration converged in %d iterations (tolerance = %.2e)\n", iter, tol);
    free_matrix(X_prev, n);

    // ---------- JACOBI METHOD ----------
    double **B = matrix(m, k);
    matrix_multiply(A, X, B, m, k, n);
    double **BT = matrix(k, m);
    transpose(B, BT, m, k);
    double **C = matrix(k, k);
    matrix_multiply(BT, B, C, k, k, m);

    double **R = matrix(k, k);
    for (int i = 0; i < k; i++)
    {
        for (int j = 0; j < k; j++)
        {
            if(i==j)
            { R[i][j] = 1;}
            else{ R[i][j] = 0; }
        }
    }
    
    // Jacobi rotation
    /*
        To diagonalize the symmetric matrix C = BT * B, whose diagonal elements 
        will approximate the eigenvalues of C (i.e., squared singular values of A).

        The algorithm iteratively finds the largest off-diagonal element and applies 
        a plane rotation to zero it out, updating both C and the accumulated 
        rotation matrix R.

        After convergence, 
            - The diagonal entries of C are the eigenvalues (λ_i)
            - The columns of R are the eigenvectors of C
            - The singular values of A are √λ_i
    
    
    */
    //using fabs to get absolute value of floating point numbers
    while (1) {
        double max = 0;
        int i_0 = 0, j_0 = 0;
        for (int i = 0; i < k; i++)
            for (int j = i + 1; j < k; j++)
                if (fabs(C[i][j]) > max) {
                    max = fabs(C[i][j]);
                    i_0 = i; j_0 = j;
                }

        if (max < 1e-10)
        { break; }

        double a_ii = C[i_0][i_0];
        double a_jj = C[j_0][j_0];
        double a_ij = C[i_0][j_0];
        double theta = 0.5 * atan2(2 * a_ij, (a_jj - a_ii));
        double c = cos(theta);
        double s = sin(theta);

        for (int i = 0; i < k; i++)
        {
            if (i != i_0 && i != j_0)
            {
                double t1 = C[i][i_0];
                double t2 = C[i][j_0];
                C[i][i_0] = c * t1 - s * t2;
                C[i][j_0] = s * t1 + c * t2;
                C[i_0][i] = C[i][i_0];
                C[j_0][i] = C[i][j_0];
            }
        }

        double new_aii = c*c*a_ii - 2*s*c*a_ij + s*s*a_jj;
        double new_ajj = s*s*a_ii + 2*s*c*a_ij + c*c*a_jj;

        C[i_0][i_0] = new_aii;
        C[j_0][j_0] = new_ajj;
        C[i_0][j_0] = 0;
        C[j_0][i_0] = 0;

        for(int i=0;i<k;i++)
        {
            double t1 = R[i][i_0];
            double t2 = R[i][j_0];
            R[i][i_0] = c * t1 - s * t2;
            R[i][j_0] = s * t1 + c * t2;
        }
    }

    // Compute Sigma
    double *Sigma = malloc(k * sizeof(double));
    for(int i=0;i<k;i++)
    {
        Sigma[i] = sqrt(fabs(C[i][i]));
    }
    // Compute V = X * R
    double **V = matrix(n, k);
    matrix_multiply(X, R, V, n, k, k);

    // Compute U = A * V * inv(Sigma)
    double **temp = matrix(m, k);
    matrix_multiply(A, V, temp, m, k, n);
    double **U = matrix(m, k);
    for (int i = 0; i < m; i++)
        for (int j = 0; j < k; j++)
            U[i][j] = (Sigma[j] > 1e-12) ? temp[i][j] / Sigma[j] : 0;

    // Reconstruct compressed image: A_k = U * Σ_k * Vᵀ
    double **Sigma_k = matrix(k, k);
    for (int i = 0; i < k; i++)
        Sigma_k[i][i] = Sigma[i];

    double **VS = matrix(k, n);
    double **VT = matrix(k, n);
    transpose(V, VT, n, k);
    matrix_multiply(Sigma_k, VT, VS, k, n, k);
    free_matrix(VT, k);

    double **A_k = matrix(m, n);
    matrix_multiply(U, VS, A_k, m, n, k);

    //code for output image
    printf("Enter output image file (PNG recommended): ");
    scanf("%s", output_image);
    write_image(output_image, A_k, m, n);
    printf("Compressed image written to %s\n", output_image);

    //Finding Frobenius Norm
    double **diff_mat = matrix(m, n);
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            diff_mat[i][j] = A[i][j] - A_k[i][j];

    double frob_error = frobenius_norm(diff_mat, m, n);
    printf("\nFrobenius norm of (A - A_k): %.6f\n", frob_error);

    //freeing the memory
    free_matrix(A, m); free_matrix(AT, n);
    free_matrix(B, m); free_matrix(BT, k);
    free_matrix(C, k); free_matrix(X, n);
    free_matrix(Y, m); free_matrix(Z, n);
    free_matrix(R, k); free_matrix(V, n);
    free_matrix(U, m); free_matrix(temp, m);
    free_matrix(Sigma_k, k); free_matrix(VS, k); free_matrix(A_k, m);
    free(Sigma);

    return 0;
}
