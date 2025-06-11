import torch

def _apply_preconditioner(preconditioner, x):
    if preconditioner is None:
        return x
    return preconditioner(x)

def _arnoldi_process(A, v1, k):
    """
    Performs k steps of the Arnoldi process.
    """
    H = torch.zeros(k + 1, k, device=v1.device, dtype=v1.dtype)
    V = torch.zeros(v1.shape[0], k + 1, device=v1.device, dtype=v1.dtype)
    V[:, 0] = v1

    for j in range(k):
        w = A(V[:, j])
        for i in range(j + 1):
            H[i, j] = torch.dot(w, V[:, i])
            w = w - H[i, j] * V[:, i]
        
        h_next = torch.linalg.norm(w)
        H[j + 1, j] = h_next
        
        # Re-orthogonalize if necessary (improves stability)
        for i in range(j + 1):
            h_re = torch.dot(w, V[:, i])
            w = w - h_re * V[:, i]
        
        H[j, j] += torch.sum(h_re)
            
        # Avoid division by zero
        if h_next > 1e-12:
            V[:, j + 1] = w / h_next
        else:
            # If h_next is zero, the Krylov subspace has been exhausted.
            # We can stop the process.
            return V[:, :j + 2], H[:j + 2, :j + 1], j + 1

    return V, H, k

def gmres(A, b, x0=None, M=None, restart=20, maxiter=None, atol=1e-5, rtol=1e-5, quiet=True):
    """
    Generalized Minimal RESidual method (GMRES) for solving Ax = b.

    Args:
        A (callable): A function that computes the matrix-vector product A(v).
        b (torch.Tensor): The right-hand side vector.
        x0 (torch.Tensor, optional): The initial guess for the solution. Defaults to zeros.
        M (callable, optional): Preconditioner for A. A function that computes M_inv(v). Defaults to None.
        restart (int, optional): Number of iterations before restarting. Defaults to 20.
        maxiter (int, optional): Maximum number of outer iterations. Defaults to None.
        atol (float, optional): Absolute tolerance for convergence. Defaults to 1e-5.
        rtol (float, optional): Relative tolerance for convergence. Defaults to 1e-5.
        quiet (bool, optional): If False, prints convergence information. Defaults to True.

    Returns:
        tuple[torch.Tensor, int]: The solution tensor and the number of iterations.
                                  A negative iteration count indicates non-convergence.
    """
    if x0 is None:
        x = torch.zeros_like(b)
    else:
        x = x0.clone().detach()

    if maxiter is None:
        maxiter = 2 * b.shape[0]

    b_norm = torch.linalg.norm(b)
    tolerance = atol + rtol * b_norm
    
    total_iters = 0

    for outer_iter in range(maxiter):
        r = _apply_preconditioner(M, b - A(x))
        r_norm = torch.linalg.norm(r)

        if not quiet:
            print(f"Outer iter {outer_iter}: residual norm = {r_norm.item():.4e}")

        if r_norm < tolerance:
            return x, total_iters

        v1 = r / r_norm
        
        V, H, actual_k = _arnoldi_process(lambda v: _apply_preconditioner(M, A(v)), v1, restart)
        
        e1 = torch.zeros(actual_k + 1, device=b.device, dtype=b.dtype)
        e1[0] = 1.0

        # Solve the least squares problem Hy = beta * e1
        y, _, _, _ = torch.linalg.lstsq(H, r_norm * e1)
        
        # Update the solution
        x = x + V[:, :actual_k] @ y
        total_iters += actual_k
        
        # Check for convergence after the inner loop
        r_final = _apply_preconditioner(M, b - A(x))
        if torch.linalg.norm(r_final) < tolerance:
            return x, total_iters

    if not quiet:
        print("GMRES did not converge within the maximum number of iterations.")
    return x, -total_iters # Negative indicates non-convergence 