function [xPhys, hist] = top3d_cf(p)
%TOP3D_CF  3-D constant-force compliant mechanism topology optimisation.
%
%   Solves
%       min_x   J = (1/|P|) sum_{k in P} [ (Fout_k - Ftar) / Ftar ]^2
%       s.t.    V(x)/V0 <= volfrac
%               R(U_k, x) = 0        (nonlinear equilibrium at every step k)
%               0 <= x <= 1
%
%   P is the plateau window (steps nRise+1 ... nStep). Steps 1..nRise are the
%   preload / rise region and are left unconstrained on purpose -- constraining
%   them would demand a force step at u = 0, which no structure can deliver.
%
%   This is NOT a modified top3d. Minimum compliance asks for the stiffest
%   layout; here we ask for a flat force-displacement response, which is a
%   different objective, needs a different optimiser, and above all needs
%   geometric nonlinearity. Under linear kinematics F = Ku, so the output
%   force is proportional to displacement and a plateau does not exist in the
%   model at all. The flat region is a large-deformation / post-buckling
%   effect. Hence: total-Lagrangian H8 elements, compressible neo-Hookean,
%   displacement-controlled Newton-Raphson, adjoint sensitivities through the
%   converged tangent.
%
%   MODEL
%       clamped : whole x = 0 face (the mount)
%       input   : line (x = Lx, y = 0,  all z), prescribed x-displacement
%       output  : line (x = Lx, y = Ly, all z), springs to ground, x-direction
%
%       Both ports sit on the free end and act along x, so they couple
%       strongly through the body: the drive pushes in at the bottom edge, the
%       jaw pushes out at the top edge. Layouts that cross directions (push in
%       x, read out in y) couple far too weakly on a uniform start -- the
%       initial force path comes out non-monotone and sign-indefinite, and the
%       optimiser has nothing to latch onto. This was measured, not guessed.
%
%   USAGE
%       top3d_cf                                  % defaults
%       q.nelx = 40; q.nelz = 8; q.Ftar = 2.0;
%       [x, hist] = top3d_cf(q);
%
%   REQUIRES  R2020b or later (pagemtimes / pagetranspose).
%
%   This file is a line-for-line translation of top3d_cf.py, whose element
%   routine and adjoint sensitivities were verified against finite differences
%   (relative error 8e-10, 9e-11 and 5e-8 respectively). Re-run that check
%   after any edit to ELEMNH: wrong sensitivities do not crash, they converge
%   quietly to the wrong shape.

% ============================== DEFAULTS =================================
d.nelx = 32; d.nely = 12; d.nelz = 6;   % elements (x = length, y = height, z = width)
d.Lx      = 0.024;   % m, domain length; elements are cubes h = Lx/nelx
d.E0      = 2.0e9;   % Pa, solid modulus (printed resin / nylon)
d.Emin    = 1e-6;    % void modulus relative to E0
d.nu      = 0.40;
d.volfrac = 0.30;
d.penal   = 3;
d.rmin    = 2.0;     % filter radius in elements
d.beta    = 1;       % Heaviside sharpness
d.betaMax = 16;
d.betaIter= 40;
d.eta     = 0.5;
d.move    = 0.05;
d.maxIter = 100;
d.uin     = -0.004;  % m, TOTAL input stroke (negative = pushing in)
d.nStep   = 10;      % load steps over that stroke
d.nRise   = 3;       % steps 1..nRise are the rise region
d.kout    = 1.0e3;   % N/m, total output spring = tissue contact stiffness
d.Ftar    = [];      % N, target plateau force; [] = auto from initial design
d.tolNR   = 1e-7;
d.maxNR   = 25;
d.plotOn  = true;

if nargin == 0, p = struct(); end
fn = fieldnames(d);
for i = 1:numel(fn)
    if ~isfield(p, fn{i}), p.(fn{i}) = d.(fn{i}); end
end
nelx = p.nelx; nely = p.nely; nelz = p.nelz;
nele = nelx*nely*nelz;  h = p.Lx/nelx;
nnode = (nelx+1)*(nely+1)*(nelz+1);  ndof = 3*nnode;

% ========================= MESH AND CONNECTIVITY =========================
% node id for 0-based (i,j,k) = (x,y,z) indices
NID = @(i,j,k) k*(nelx+1)*(nely+1) + i*(nely+1) + j + 1;
[jj, ii, kk] = ndgrid(0:nely-1, 0:nelx-1, 0:nelz-1);   % j fastest
jj = jj(:); ii = ii(:); kk = kk(:);                    % matches e = (k)*nelx*nely+(i)*nely+j+1
cor = [NID(ii,jj,kk)     NID(ii+1,jj,kk)     NID(ii+1,jj+1,kk)     NID(ii,jj+1,kk) ...
       NID(ii,jj,kk+1)   NID(ii+1,jj,kk+1)   NID(ii+1,jj+1,kk+1)   NID(ii,jj+1,kk+1)];
edofMat = zeros(nele,24);
for a = 1:8
    edofMat(:,3*a-2:3*a) = [3*cor(:,a)-2, 3*cor(:,a)-1, 3*cor(:,a)];
end
% MATLAB reshape is column-major, so the flat index of a (24 x 24 x nele)
% page array permuted to (nele x 24 x 24) is c = (J-1)*24 + I. Row index is
% therefore the FAST one. (This is the opposite of numpy -- easy to get wrong.)
rowIdx = repmat( edofMat, 1, 24);   % col c -> edofMat(:, mod(c-1,24)+1) = I
colIdx = repelem(edofMat, 1, 24);   % col c -> edofMat(:, ceil(c/24))    = J

% ---------------------------- PROBLEM SET-UP -----------------------------
[jc, kc] = ndgrid(0:nely, 0:nelz);
clampN  = NID(zeros(numel(jc),1), jc(:), kc(:));
fixdof  = reshape([3*clampN-2, 3*clampN-1, 3*clampN].', [], 1);
kline   = (0:nelz)';
indofs  = 3*NID(nelx*ones(nelz+1,1), zeros(nelz+1,1),      kline) - 2;
outdofs = 3*NID(nelx*ones(nelz+1,1), nely*ones(nelz+1,1),  kline) - 2;
ks      = p.kout/numel(outdofs);                 % spring per output node
freedof = setdiff((1:ndof)', [fixdof; indofs]);
nfree   = numel(freedof);
dofmap  = zeros(ndof,1);  dofmap(freedof) = 1:nfree;
outFr   = dofmap(outdofs);

keepIdx = (dofmap(rowIdx(:)) > 0) & (dofmap(colIdx(:)) > 0);
Kr = [dofmap(rowIdx(keepIdx)); outFr];           % springs share the pattern
Kc = [dofmap(colIdx(keepIdx)); outFr];

% ===================== ELEMENT PRE-COMPUTATION (once) ====================
xa = [-1 1 1 -1 -1 1 1 -1];  ya = [-1 -1 1 1 -1 -1 1 1];  za = [-1 -1 -1 -1 1 1 1 1];
g  = 1/sqrt(3);
dNdX_all = zeros(3,8,8);  wdet = zeros(8,1);  gp = 0;
Xe = 0.5*h*[xa+1; ya+1; za+1].';
for xi = [-g g]
    for et = [-g g]
        for ze = [-g g]
            gp = gp + 1;
            dN = 0.125*[ xa.*(1+et*ya).*(1+ze*za);
                         ya.*(1+xi*xa).*(1+ze*za);
                         za.*(1+xi*xa).*(1+et*ya) ];
            J0 = dN*Xe;
            dNdX_all(:,:,gp) = J0\dN;
            wdet(gp) = abs(det(J0));             % Gauss weights are 1
        end
    end
end
mu1  = 1/(2*(1+p.nu));                           % neo-Hookean constants at E = 1
lam1 = p.nu/((1+p.nu)*(1-2*p.nu));

% ============================ DENSITY FILTER =============================
% Built by offsets rather than a per-element neighbour loop -- same result,
% orders of magnitude faster on 3-D meshes.
eidx = reshape(1:nele, nely, nelx, nelz);
R = ceil(p.rmin) - 1;  Hi = []; Hj = []; Hv = [];
for dk = -R:R
    for di = -R:R
        for dj = -R:R
            dd = sqrt(di^2 + dj^2 + dk^2);
            if dd >= p.rmin, continue; end
            sj = (1+max(0,-dj)):(nely-max(0,dj));  tj = (1+max(0,dj)):(nely-max(0,-dj));
            si = (1+max(0,-di)):(nelx-max(0,di));  ti = (1+max(0,di)):(nelx-max(0,-di));
            sk = (1+max(0,-dk)):(nelz-max(0,dk));  tk = (1+max(0,dk)):(nelz-max(0,-dk));
            src = eidx(sj,si,sk); tgt = eidx(tj,ti,tk);
            Hi = [Hi; tgt(:)]; Hj = [Hj; src(:)];
            Hv = [Hv; (p.rmin-dd)*ones(numel(src),1)]; %#ok<AGROW>
        end
    end
end
H = sparse(Hi, Hj, Hv, nele, nele);  Hs = sum(H,2);

% ============================== INITIALISE ===============================
x = p.volfrac*ones(nele,1);  beta = p.beta;  Ftar = p.Ftar;
Upath = zeros(ndof, p.nStep);                    % warm-start states
hist = struct('J',[],'vol',[],'ripple',[],'Fout',[]);
uAxis = abs(p.uin)*(1:p.nStep)'/p.nStep;
fprintf('mesh %dx%dx%d = %d elements, %d dof (%d free)\n', ...
        nelx, nely, nelz, nele, ndof, nfree);
fprintf('%4s %11s %7s %9s %8s %8s %7s\n', ...
        'it','J','vol','Fmean','ripple%','change','s/it');

% ============================= MAIN LOOP =================================
for iter = 1:p.maxIter
    tic;
    xTilde = (H*x)./Hs;
    [xPhys, dxdxt] = project(xTilde, beta, p.eta);
    Eabs  = p.E0*(p.Emin + xPhys.^p.penal*(1-p.Emin));
    dEabs = p.E0*p.penal*xPhys.^(p.penal-1)*(1-p.Emin);

    [Fout, dFout, ok] = forcePath(Eabs, dEabs);
    if ~ok
        warning('Newton failed at iteration %d. Reduce move or uin.', iter);
        break
    end

    P = (p.nRise+1):p.nStep;
    if isempty(Ftar)
        Ftar = median(Fout(P));
        if abs(Ftar) < 1e-3 || any(sign(Fout(P)) ~= sign(Ftar))
            error(['Initial plateau window is degenerate (Fout = %s). The ' ...
                   'ports are too weakly coupled or the stroke is too small; ' ...
                   'fix the load case before optimising.'], mat2str(Fout',3));
        end
        fprintf('  auto target Ftar = %.3f N (median plateau of uniform start)\n', Ftar);
    end

    r  = (Fout(P) - Ftar)/Ftar;  nP = numel(r);
    J  = (r'*r)/nP;
    dJ = dFout(:,P)*(2*r/(Ftar*nP));             % d/d xPhys
    dJ = H*((dJ.*dxdxt)./Hs);                    % chain rule through the filter
    dV = H*(dxdxt./Hs)/nele;

    % ---- design update: projected gradient + bisection on the volume mult.
    % Optimality Criteria cannot be used here: it assumes single-signed
    % sensitivities, and these flip sign depending on whether a load step sits
    % above or below the target. Swap in Svanberg's mmasub for real runs.
    dJn = dJ/max(max(abs(dJ)), eps);
    dVn = dV/max(max(abs(dV)), eps);
    lo = -1e4; hi = 1e4;
    while hi - lo > 1e-6
        mid  = 0.5*(lo+hi);
        xnew = min(1, max(0, min(x+p.move, max(x-p.move, x - p.move*(dJn + mid*dVn)))));
        xp   = project((H*xnew)./Hs, beta, p.eta);
        if mean(xp) > p.volfrac, lo = mid; else, hi = mid; end
    end
    change = max(abs(xnew - x));  x = xnew;

    Fp = Fout(P);  ripple = 100*(max(Fp)-min(Fp))/abs(mean(Fp));
    fprintf('%4d %11.4e %7.3f %9.3f %8.2f %8.4f %7.1f\n', ...
            iter, J, mean(xPhys), mean(Fp), ripple, change, toc);
    hist.J(end+1) = J;  hist.vol(end+1) = mean(xPhys);
    hist.ripple(end+1) = ripple;  hist.Fout(:,end+1) = Fout;
    if p.plotOn, drawState(); end

    if mod(iter, p.betaIter) == 0 && beta < p.betaMax
        beta = 2*beta;  change = 1;
        fprintf('  beta -> %g\n', beta);
    end
    if change < 0.002 && beta >= p.betaMax, break; end
end

% ======================== NESTED HELPER FUNCTIONS ========================
    function [Fout, dFout, ok] = forcePath(Eabs, dEabs)
        % Walk the load path. States are kept in Upath and reused as initial
        % guesses on the next call -- between optimisation iterations the
        % design barely moves, which cuts Newton to two or three iterations
        % per step after the first pass (measured: ~6x overall speed-up).
        Fout = zeros(p.nStep,1);  dFout = zeros(nele, p.nStep);  ok = true;
        rhs = zeros(nfree,1);  rhs(outFr) = ks;
        U = zeros(ndof,1);  dc = [];
        for kstep = 1:p.nStep
            target = p.uin*kstep/p.nStep;
            needLU = (nargin > 1) && kstep > p.nRise;
            [Uk, dc2, fhat, good] = stepSolve(Upath(:,kstep), target, Eabs, dc, needLU);
            if ~good     % warm start failed -> fall back to continuation
                [Uk, dc2, fhat, good] = stepSolve(U, target, Eabs, [], needLU);
                if ~good, ok = false; return; end
            end
            U = Uk;  dc = dc2;  Upath(:,kstep) = U;
            Fout(kstep) = ks*sum(U(outdofs));
            if needLU
                lam = zeros(ndof,1);  lam(freedof) = dc\rhs;
                dFout(:,kstep) = -dEabs .* sum(lam(edofMat).*fhat, 2);
            end
        end
    end

    function [U, dc, fhat, ok] = stepSolve(U0, target, Eabs, dc, needLU)
        [U, dc2, fhat, ok] = newton(U0, target, Eabs, dc, needLU);
        if ok, dc = dc2; return; end
        if ~isempty(dc)                                  % stale factorisation?
            [U, dc2, fhat, ok] = newton(U0, target, Eabs, [], needLU);
            if ok, dc = dc2; return; end
        end
        u0 = U0(indofs(1));  nsub = 2;                   % sub-step the increment
        while nsub <= 8
            Ut = U0;  okAll = true;
            for s = 1:nsub
                [Ut, dc2, fhat, good] = newton(Ut, u0 + (target-u0)*s/nsub, ...
                                               Eabs, [], needLU);
                if ~good, okAll = false; break; end
            end
            if okAll, U = Ut; dc = dc2; ok = true; return; end
            nsub = 2*nsub;
        end
        U = U0;  ok = false;
    end

    function [U, dc, fhat, ok] = newton(U, target, Eabs, dc, needLU)
        % Modified Newton. Factorisation dominates the cost (roughly 0.46 s vs
        % 0.11 s for a tangent assembly on the default mesh), so it is carried
        % across Newton iterations and across load steps, and only rebuilt
        % when the residual stops dropping. A fresh factorisation is forced at
        % convergence of a plateau step: the adjoint needs the exact tangent.
        ok = false;  fhat = [];
        refac = isempty(dc);  nRprev = inf;
        U(indofs) = target;
        for it = 1:p.maxNR
            if refac
                [fhat, khat] = elemNH(U(edofMat));
            else
                fhat = elemNH(U(edofMat));
            end
            Fint = accumarray(edofMat(:), reshape(fhat.*Eabs,[],1), [ndof 1]);
            Fint(outdofs) = Fint(outdofs) + ks*U(outdofs);
            Rf = Fint(freedof);  nR = norm(Rf);
            if refac, dc = decomposition(tangent(khat, Eabs)); end
            if nR < p.tolNR*max(1, norm(Fint))
                if needLU && ~refac                      % exact tangent needed
                    [fhat, khat] = elemNH(U(edofMat));
                    dc = decomposition(tangent(khat, Eabs));
                end
                ok = true; return
            end
            dU = -(dc\Rf);
            alpha = 1;                                   % step-halving search
            for ls = 1:6
                Ut = U;  Ut(freedof) = U(freedof) + alpha*dU;
                f2 = elemNH(Ut(edofMat));
                F2 = accumarray(edofMat(:), reshape(f2.*Eabs,[],1), [ndof 1]);
                F2(outdofs) = F2(outdofs) + ks*Ut(outdofs);
                if norm(F2(freedof)) < nR || alpha < 0.05, break; end
                alpha = alpha/2;
            end
            U(freedof) = U(freedof) + alpha*dU;
            refac = (nR > 0.35*nRprev);                  % poor reduction
            nRprev = nR;
        end
    end

    function Kff = tangent(khat, Eabs)
        vals = khat .* reshape(Eabs, 1, 1, nele);
        vals = reshape(permute(vals, [3 1 2]), nele, 576);
        vals = [vals(keepIdx); ks*ones(numel(outFr),1)];
        Kff  = sparse(Kr, Kc, vals, nfree, nfree);
    end

    function [fhat, khat] = elemNH(Ue)
        % Unit-modulus neo-Hookean H8, vectorised over all elements.
        % Ue (nele x 24) -> fhat (nele x 24), khat (24 x 24 x nele).
        wantK = (nargout > 1);
        fhat  = zeros(nele,24);
        if wantK, khat = zeros(24,24,nele); else, khat = []; end
        VI = [1 1; 2 2; 3 3; 1 2; 2 3; 1 3];          % Voigt -> (i,j)
        cmp = {Ue(:,1:3:24), Ue(:,2:3:24), Ue(:,3:3:24)};   % each nele x 8
        F = cell(3,3); C = cell(3,3); Ci = cell(3,3); S = cell(3,3);
        for gq = 1:8
            dNdX = dNdX_all(:,:,gq);  w = wdet(gq);
            for a = 1:3
                for b = 1:3
                    F{a,b} = cmp{a}*dNdX(b,:)' + (a==b);   % F(e,a,b)
                end
            end
            for a = 1:3
                for b = a:3
                    C{a,b} = F{1,a}.*F{1,b} + F{2,a}.*F{2,b} + F{3,a}.*F{3,b};
                    C{b,a} = C{a,b};
                end
            end
            detC = C{1,1}.*(C{2,2}.*C{3,3} - C{2,3}.^2) ...
                 - C{1,2}.*(C{1,2}.*C{3,3} - C{2,3}.*C{1,3}) ...
                 + C{1,3}.*(C{1,2}.*C{2,3} - C{2,2}.*C{1,3});
            detC = max(detC, 1e-8);                 % guard on inverted voids
            lnJ  = 0.5*log(detC);
            Ci{1,1} = (C{2,2}.*C{3,3} - C{2,3}.^2)./detC;
            Ci{2,2} = (C{1,1}.*C{3,3} - C{1,3}.^2)./detC;
            Ci{3,3} = (C{1,1}.*C{2,2} - C{1,2}.^2)./detC;
            Ci{1,2} = (C{1,3}.*C{2,3} - C{1,2}.*C{3,3})./detC;  Ci{2,1} = Ci{1,2};
            Ci{1,3} = (C{1,2}.*C{2,3} - C{1,3}.*C{2,2})./detC;  Ci{3,1} = Ci{1,3};
            Ci{2,3} = (C{1,3}.*C{1,2} - C{1,1}.*C{2,3})./detC;  Ci{3,2} = Ci{2,3};
            for a = 1:3
                for b = 1:3
                    S{a,b} = mu1*((a==b) - Ci{a,b}) + lam1*lnJ.*Ci{a,b};
                end
            end
            Br = cell(6,1);
            for I = 1:6
                i1 = VI(I,1); j1 = VI(I,2);
                Br{I} = zeros(nele,24);
                for k = 1:3
                    val = F{k,i1}*dNdX(j1,:);
                    if i1 ~= j1, val = val + F{k,j1}*dNdX(i1,:); end
                    Br{I}(:,k:3:24) = val;
                end
            end
            for I = 1:6
                fhat = fhat + w*(Br{I}.*S{VI(I,1),VI(I,2)});
            end
            if ~wantK, continue; end
            c = 2*(mu1 - lam1*lnJ);
            Dp = zeros(6,6,nele);
            for I = 1:6
                i1 = VI(I,1); j1 = VI(I,2);
                for Jq = 1:6
                    k1 = VI(Jq,1); l1 = VI(Jq,2);
                    Dp(I,Jq,:) = lam1*Ci{i1,j1}.*Ci{k1,l1} ...
                        + c.*0.5.*(Ci{i1,k1}.*Ci{j1,l1} + Ci{i1,l1}.*Ci{j1,k1});
                end
            end
            Bp = zeros(6,24,nele);
            for I = 1:6, Bp(I,:,:) = Br{I}.'; end
            khat = khat + w*pagemtimes(pagetranspose(Bp), pagemtimes(Dp, Bp));
            for a = 1:8                                  % geometric stiffness
                for b = 1:8
                    gg = zeros(nele,1);
                    for i1 = 1:3
                        for j1 = 1:3
                            gg = gg + dNdX(i1,a)*dNdX(j1,b)*S{i1,j1};
                        end
                    end
                    gg = reshape(w*gg, 1, 1, nele);
                    for k = 1:3
                        khat(3*(a-1)+k, 3*(b-1)+k, :) = ...
                            khat(3*(a-1)+k, 3*(b-1)+k, :) + gg;
                    end
                end
            end
        end
    end

    function drawState()
        subplot(2,1,1); cla;
        rho = permute(reshape(xPhys, nely, nelx, nelz), [2 1 3]);  % x,y,z
        pv = patch(isosurface(permute(rho,[2 1 3]), 0.5));
        set(pv,'FaceColor',[0.4 0.5 0.7],'EdgeColor','none');
        axis equal tight; view(35,25); camlight; lighting gouraud; box on;
        title(sprintf('iter %d   \\beta = %g', iter, beta));
        subplot(2,1,2); cla;
        plot(1e3*uAxis, Fout, '-o', 'LineWidth', 1.2); hold on
        plot(1e3*uAxis([1 end]), [Ftar Ftar], 'r--');
        yl = [0 1.2*max(max(Fout), Ftar)];
        plot(1e3*uAxis(p.nRise)*[1 1], yl, 'k:');
        ylim(yl); grid on
        xlabel('input stroke (mm)'); ylabel('F_{out} (N)');
        legend('mechanism','target','plateau starts','Location','SouthEast');
        drawnow;
    end
end

% ------------------------------------------------------------------------
function [xPhys, dx] = project(xTilde, beta, eta)
tb  = tanh(beta*eta);
den = tb + tanh(beta*(1-eta));
xPhys = (tb + tanh(beta*(xTilde-eta)))/den;
dx    = beta*(1 - tanh(beta*(xTilde-eta)).^2)/den;
end
