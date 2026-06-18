import sys, math, random
import numpy as np
import cocoex

def main():
    lines = [l.strip() for l in sys.stdin.readlines()]
    if len(lines) < 6:
        return
    suite_name = lines[0]
    func_idx = int(lines[1])
    dim = int(lines[2])
    inst = int(lines[3])
    budget = int(lines[4])
    seed = int(lines[5])

    random.seed(seed)
    np.random.seed(seed)

    suite = cocoex.Suite(
        suite_name=suite_name,
        suite_instance="year: 2009",
        suite_options=(
            f"function_indices: {func_idx} "
            f"instance_indices: {inst} "
            f"dimensions: {dim}"
        ),
    )
    problem = next(iter(suite))
    lb = np.array(problem.lower_bounds, dtype=float)
    ub = np.array(problem.upper_bounds, dtype=float)

    mean = (lb + ub) / 2.0
    sigma = 0.3 * np.mean(ub - lb)

    lam = 4 + int(3 * math.log(dim))
    mu = lam // 2
    weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    weights /= weights.sum()
    mueff = 1.0 / np.sum(weights ** 2)

    cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
    cs = (mueff + 2) / (dim + mueff + 5)
    c1 = 2 / ((dim + 1.3) ** 2 + mueff)
    cmu = min(1 - c1, 2 * (mueff - 2 + 1) / ((dim + 2) ** 2 + mueff))
    damps = 1 + 2 * max(0, math.sqrt((mueff - 1) / (dim + 1)) - 1) + cs

    pc = np.zeros(dim)
    ps = np.zeros(dim)
    B = np.eye(dim)
    D = np.ones(dim)
    C = B @ np.diag(D ** 2) @ B.T
    invsqrtC = B @ np.diag(D ** -1) @ B.T
    eigen_updated = 0
    counteval = 0

    best_x = mean.copy()
    best_f = float('inf')

    while counteval < budget:
        lam_eff = min(lam, budget - counteval)
        arz = np.random.randn(lam_eff, dim)
        ary = arz @ np.diag(D) @ B.T
        arx = mean + sigma * ary
        np.clip(arx, lb, ub, out=arx)

        fitness = np.empty(lam_eff)
        for i in range(lam_eff):
            fitness[i] = problem(arx[i].tolist())
        counteval += lam_eff

        idx = np.argsort(fitness)
        if fitness[idx[0]] < best_f:
            best_f = fitness[idx[0]]
            best_x = arx[idx[0]].copy()

        xold = mean.copy()
        mean = np.dot(weights, arx[idx[:mu]])

        y = mean - xold
        z = (1 / sigma) * invsqrtC @ y
        ps = (1 - cs) * ps + math.sqrt(cs * (2 - cs) * mueff) * z
        hsig = int((np.linalg.norm(ps) / math.sqrt(1 - (1 - cs) ** (2 * counteval / lam))) < (1.4 + 2 / (dim + 1)))
        pc = (1 - cc) * pc + hsig * math.sqrt(cc * (2 - cc) * mueff) * y / sigma

        C = (1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C)
        artmp = ary[idx[:mu]].T
        C += cmu * artmp @ np.diag(weights) @ artmp.T

        sigma *= math.exp((cs / damps) * (np.linalg.norm(ps) / math.sqrt(dim) - 1))

        if counteval - eigen_updated > lam / (c1 + cmu) / dim / 10:
            eigen_updated = counteval
            D, B = np.linalg.eigh(C)
            D = np.sqrt(np.maximum(D, 1e-20))
            invsqrtC = B @ np.diag(D ** -1) @ B.T

    print(' '.join(str(v) for v in best_x.tolist()))

if __name__ == "__main__":
    main()
