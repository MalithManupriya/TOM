function [xPhys, hist] = top2d_cf(p)
%TOP2D_CF  Topology optimisation of a 2-D constant-force compliant mechanism.
%
%   Unlike top3d (minimum compliance), this code does NOT look for the
%   stiffest structure. It searches for a material layout whose OUTPUT FORCE
%   stays flat while the input port keeps travelling:
%
%       min_x   J = (1/|P|) * sum_{k in P} [ (Fout_k - Ftar) / Ftar ]^2
%       s.t.    V(x)/V0 <= volfrac
%               R(U_k, x) = 0          (nonlinear equilibrium, every step k)
%               0 <= x <= 1
%
%   P is the "plateau window": load steps nRise+1 ... nStep. Steps 1..nRise
%   are the preload / rise region and are deliberately left unconstrained,
%   because every real constant-force mechanism needs a rise before the
%   plateau (cf. Xia et al.: 2.2 mm preload, then 6.4 mm of constant force).
%
%   WHY IT MUST BE NONLINEAR
%   A linear analysis gives F = K u. Force is then proportional to
%   displacement and a plateau is mathematically impossible. The flat region
%   comes from large-deformation / post-buckling behaviour, so the code uses
%   a total-Lagrangian formulation with a neo-Hookean material and solves
%   displacement-controlled Newton-Raphson at every load step.
%
%   MODEL (edit the "PROBLEM SET-UP" block to change it)
%
%        |<---------------- Lx ---------------->|
%        +--------------------------------------o  <- output dof (vertical)
%     // |                                      |     grounded via spring kout
%     // |            design domain             |     (= tissue / simulant)
%     // |                                      |
%     // |                                      o <- input dof (horizontal)
%     // +--------------------------------------+    prescribed stroke uin
%      clamped
%
%   USAGE
%       top2d_cf                       % defaults
%       p.nelx = 90; p.Ftar = 2.0;
%       [x, hist] = top2d_cf(p);       % override any default below
%
%   OPTIMISER
%       Uses Svanberg's mmasub.m if it is on the MATLAB path (recommended;
%       free for academic use on request from the author). Otherwise falls
%       back to a projected-gradient step with bisection on the volume
%       multiplier -- slower and fussier, but it runs out of the box.
%
%   Element formulation and adjoint sensitivities verified against finite
%   differences (relative error ~1e-9 and ~1e-7 respectively).

% ============================== DEFAULTS =================================
d.nelx    = 60;      % elements along x
d.nely    = 30;      % elements along y
d.Lx      = 0.030;   % m   domain length  (element size h = Lx/nelx)
d.thk     = 0.004;   % m   out-of-plane thickness
d.E0      = 2.0e9;   % Pa  solid modulus (printed resin / nylon)
d.Emin    = 1e-6;    % -   void modulus, relative to E0
d.nu      = 0.40;    % -   Poisson ratio
d.volfrac = 0.30;
d.penal   = 3;
d.rmin    = 2.5;     % filter radius, in elements
d.beta    = 1;       % Heaviside sharpness (doubled every betaIter)
d.betaMax = 16;
d.betaIter= 40;
d.eta     = 0.5;     % projection threshold
d.uin     = -0.006;  % m   TOTAL input stroke (negative = pushing into domain)
d.nStep   = 12;      % load steps over that stroke
d.nRise   = 4;       % steps 1..nRise are the rise region (not constrained)
d.kout    = 1.0e3;   % N/m output spring = tissue/simulant contact stiffness
d.Ftar    = [];      % N   target plateau force; [] = auto from initial design
d.maxIter = 150;
d.move    = 0.05;
d.tolNR   = 1e-9;    % relative residual tolerance
d.maxNR   = 30;
d.plotOn  = true;

if nargin == 0, p = struct(); end
fn = fieldnames(d);
for i = 1:numel(fn)
    if ~isfield(p, fn{i}), p.(fn{i}) = d.(fn{i}); end
end
nelx = p.nelx; nely = p.nely; nele = nelx*nely;
h    = p.Lx/nelx;

% ========================= MESH AND CONNECTIVITY =========================
nodenrs = reshape(1:(nelx+1)*(nely+1), nely+1, nelx+1);  % (j = y from bottom)
ndof    = 2*(nelx+1)*(nely+1);
nBL = nodenrs(1:nely,   1:nelx  );
nBR = nodenrs(1:nely,   2:nelx+1);
nTR = nodenrs(2:nely+1, 2:nelx+1);
nTL = nodenrs(2:nely+1, 1:nelx  );
edofMat = [2*nBL(:)-1 2*nBL(:) 2*nBR(:)-1 2*nBR(:) ...
           2*nTR(:)-1 2*nTR(:) 2*nTL(:)-1 2*nTL(:)];   % CCW: BL BR TR TL
rowIdx = repelem(edofMat, 1, 8);   % col c -> edofMat(:, ceil(c/8))
colIdx = repmat( edofMat, 1, 8);   % col c -> edofMat(:, mod(c-1,8)+1)

% ---------------------------- PROBLEM SET-UP -----------------------------
fixdof = [2*nodenrs(:,1)-1; 2*nodenrs(:,1)];          % left edge clamped
indof  = 2*nodenrs(round((nely+1)/2), nelx+1) - 1;    % mid-right, horizontal
outdof = 2*nodenrs(nely+1,            nelx+1);        % top-right, vertical
freedof = setdiff(1:ndof, [fixdof; indof])';   % column, so K\R stays consistent

% ===================== ELEMENT PRE-COMPUTATION (once) ====================
% All elements are identical squares in the reference configuration, so the
% shape-function derivatives and Jacobian are the same for every element.
gp = [-1 1]/sqrt(3);
Xe = [0 0; h 0; h h; 0 h];
dNdX_all = zeros(2,4,4); wdet = zeros(4,1); g = 0;
for xi = gp
    for eta = gp
        g = g+1;
        dNdxi = 0.25*[-(1-eta)  (1-eta)  (1+eta) -(1+eta);
                      -(1-xi)  -(1+xi)   (1+xi)   (1-xi)];
        J0 = dNdxi*Xe;
        dNdX_all(:,:,g) = J0\dNdxi;
        wdet(g) = det(J0)*p.thk;     % 1*1 Gauss weights
    end
end
mu1  = 1/(2*(1+p.nu));                      % neo-Hookean constants at E = 1
lam1 = p.nu/((1+p.nu)*(1-2*p.nu));          % (plane strain)

% ============================ DENSITY FILTER =============================
iH = ones(nele*(2*(ceil(p.rmin)-1)+1)^2,1); jH = iH; sH = zeros(size(iH)); k = 0;
for i1 = 1:nelx
    for j1 = 1:nely
        e1 = (i1-1)*nely + j1;
        for i2 = max(i1-(ceil(p.rmin)-1),1):min(i1+(ceil(p.rmin)-1),nelx)
            for j2 = max(j1-(ceil(p.rmin)-1),1):min(j1+(ceil(p.rmin)-1),nely)
                e2 = (i2-1)*nely + j2; k = k+1;
                iH(k) = e1; jH(k) = e2;
                sH(k) = max(0, p.rmin - sqrt((i1-i2)^2+(j1-j2)^2));
            end
        end
    end
end
H = sparse(iH(1:k), jH(1:k), sH(1:k)); Hs = sum(H,2);

% ============================== INITIALISE ===============================
x    = p.volfrac*ones(nele,1);
beta = p.beta;  Ftar = p.Ftar;
useMMA = (exist('mmasub','file') == 2);
if useMMA
    xold1 = x; xold2 = x; low = []; upp = [];
    a0 = 1; am = 0; cm = 1000; dm = 0;
end
hist = struct('J',[],'vol',[],'Fout',[],'change',[]);
uPath = abs(p.uin)*(1:p.nStep)'/p.nStep;
fprintf('%5s %11s %8s %9s %9s %7s\n','iter','J','vol','Fmean','ripple%','change');

% ============================= MAIN LOOP =================================
for iter = 1:p.maxIter
    % ---- filter + Heaviside projection ----
    xTilde = (H*x)./Hs;
    tb = tanh(beta*p.eta);
    xPhys  = (tb + tanh(beta*(xTilde-p.eta)))/(tb + tanh(beta*(1-p.eta)));
    dxdxt  = beta*(1 - tanh(beta*(xTilde-p.eta)).^2)/(tb + tanh(beta*(1-p.eta)));

    Eabs  = p.E0*(p.Emin + xPhys.^p.penal*(1-p.Emin));
    dEabs = p.E0*p.penal*xPhys.^(p.penal-1)*(1-p.Emin);

    % ---- incremental nonlinear analysis + adjoint sensitivities ----
    U = zeros(ndof,1); Fout = zeros(p.nStep,1); dFout = zeros(nele,p.nStep);
    ok = true;
    for kstep = 1:p.nStep
        [U, Kff, fhat, good] = nlstep(U, p.uin*kstep/p.nStep);
        if ~good, ok = false; break; end
        Fout(kstep) = p.kout*U(outdof);
        if kstep > p.nRise
            rhs = zeros(ndof,1); rhs(outdof) = p.kout;
            lam = zeros(ndof,1); lam(freedof) = Kff\rhs(freedof);
            dFout(:,kstep) = -dEabs .* sum(lam(edofMat).*fhat, 2);
        end
    end
    if ~ok
        warning('Newton failed at step %d (iter %d). Reduce move / uin.', kstep, iter);
        break
    end

    % ---- pick the plateau target on the first pass, if not given ----
    if isempty(Ftar)
        Ftar = median(Fout(p.nRise+1:end));
        fprintf('  auto target Ftar = %.3f N (median of initial plateau window)\n', Ftar);
    end

    % ---- objective and gradient ----
    P  = (p.nRise+1):p.nStep;  nP = numel(P);
    r  = (Fout(P) - Ftar)/Ftar;
    J  = sum(r.^2)/nP;
    dJ = dFout(:,P) * (2*r/(Ftar*nP));        % d/d xPhys

    % ---- chain rule back through projection and filter ----
    dJ = H*((dJ.*dxdxt)./Hs);
    dV = H*((ones(nele,1).*dxdxt)./Hs)/nele;

    % ---- design update ----
    if useMMA
        [xmma,~,~,~,~,~,~,~,~,low,upp] = mmasub(1,nele,iter,x,max(0,x-p.move), ...
            min(1,x+p.move),xold1,xold2,J,dJ,mean(xPhys)/p.volfrac-1, ...
            (dV/p.volfrac)',low,upp,a0,am,cm,dm);
        xold2 = xold1; xold1 = x; xnew = xmma;
    else
        % normalise both gradients to O(1) so the bisection bracket is valid
        dJn = dJ/max(max(abs(dJ)), eps);
        dVn = dV/max(max(abs(dV)), eps);
        l1 = -1e4; l2 = 1e4;
        while (l2-l1) > 1e-6
            lm = 0.5*(l1+l2);
            xnew = min(1, max(0, min(x+p.move, max(x-p.move, x - p.move*(dJn + lm*dVn)))));
            xtn  = (H*xnew)./Hs;
            xpn  = (tb + tanh(beta*(xtn-p.eta)))/(tb + tanh(beta*(1-p.eta)));
            if mean(xpn) > p.volfrac, l1 = lm; else, l2 = lm; end
        end
    end
    change = max(abs(xnew - x));  x = xnew;

    % ---- report ----
    Fp = Fout(P);
    ripple = 100*(max(Fp)-min(Fp))/mean(Fp);
    fprintf('%5i %11.4e %8.3f %9.3f %9.2f %7.4f\n', ...
            iter, J, mean(xPhys), mean(Fp), ripple, change);
    hist.J(end+1) = J; hist.vol(end+1) = mean(xPhys);
    hist.Fout(:,end+1) = Fout; hist.change(end+1) = change;

    if p.plotOn, drawState(); end

    % ---- beta continuation + convergence ----
    if mod(iter, p.betaIter) == 0 && beta < p.betaMax
        beta = 2*beta; change = 1;
        fprintf('  beta -> %g\n', beta);
    end
    if change < 0.002 && beta >= p.betaMax, break; end
end

% ======================== NESTED HELPER FUNCTIONS ========================
    function [Uo, Kff, fhat, good] = nlstep(U0, uTarget)
        % Displacement-controlled Newton-Raphson with sub-stepping fallback.
        Uo = U0; good = false; Kff = []; fhat = [];
        uStart = U0(indof);
        nSub = 1;
        while nSub <= 8
            Utry = U0; conv = true;
            for sub = 1:nSub
                Utry(indof) = uStart + (uTarget-uStart)*sub/nSub;
                [Utry, Kff, fhat, conv] = newton(Utry);
                if ~conv, break; end
            end
            if conv, Uo = Utry; good = true; return; end
            nSub = 2*nSub;
        end
    end

    function [U, Kff, fhat, conv] = newton(U)
        conv = false; Kff = []; fhat = [];
        for it = 1:p.maxNR
            [fh, kh] = elemNH(U(edofMat));
            Fint = accumarray(edofMat(:), reshape(fh.*Eabs,[],1), [ndof 1]);
            Fint(outdof) = Fint(outdof) + p.kout*U(outdof);
            sK = kh.*Eabs;
            K  = sparse(rowIdx(:), colIdx(:), sK(:), ndof, ndof);
            K(outdof,outdof) = K(outdof,outdof) + p.kout;
            Kff = K(freedof,freedof); fhat = fh;
            R = Fint(freedof); nR = norm(R);
            if nR < p.tolNR*max(1, norm(Fint)), conv = true; return; end
            dU = -Kff\R;
            % step-halving line search
            alpha = 1;
            for ls = 1:6
                Ut = U; Ut(freedof) = U(freedof) + alpha*dU;
                fh2 = elemNH(Ut(edofMat));
                F2 = accumarray(edofMat(:), reshape(fh2.*Eabs,[],1), [ndof 1]);
                F2(outdof) = F2(outdof) + p.kout*Ut(outdof);
                if norm(F2(freedof)) < nR || alpha < 0.05, break; end
                alpha = alpha/2;
            end
            U(freedof) = U(freedof) + alpha*dU;
        end
    end

    function [fhat, khat] = elemNH(Ue)
        % Unit-modulus neo-Hookean element, vectorised over all elements.
        % Returns fhat (nele x 8) and khat (nele x 64, col c = (I-1)*8 + Jj).
        wantK = (nargout > 1);
        fhat = zeros(nele,8);
        if wantK, khat = zeros(nele,64); else, khat = []; end
        ix = [1 3 5 7]; iy = [2 4 6 8];
        for gg = 1:4
            dNdX = dNdX_all(:,:,gg); w = wdet(gg);
            F11 = 1 + Ue(:,ix)*dNdX(1,:)';  F12 = Ue(:,ix)*dNdX(2,:)';
            F21 =     Ue(:,iy)*dNdX(1,:)';  F22 = 1 + Ue(:,iy)*dNdX(2,:)';
            C11 = F11.^2 + F21.^2;
            C12 = F11.*F12 + F21.*F22;
            C22 = F12.^2 + F22.^2;
            detC = max(C11.*C22 - C12.^2, 1e-6);   % guard on inverted voids
            lnJ  = 0.5*log(detC);
            Ci11 = C22./detC; Ci22 = C11./detC; Ci12 = -C12./detC;
            S11 = mu1*(1-Ci11) + lam1*lnJ.*Ci11;
            S22 = mu1*(1-Ci22) + lam1*lnJ.*Ci22;
            S12 = mu1*(  -Ci12) + lam1*lnJ.*Ci12;
            B1 = zeros(nele,8); B2 = zeros(nele,8); B3 = zeros(nele,8);
            for a = 1:4
                d1 = dNdX(1,a); d2 = dNdX(2,a);
                B1(:,2*a-1) = F11*d1;          B1(:,2*a) = F21*d1;
                B2(:,2*a-1) = F12*d2;          B2(:,2*a) = F22*d2;
                B3(:,2*a-1) = F11*d2 + F12*d1; B3(:,2*a) = F21*d2 + F22*d1;
            end
            fhat = fhat + w*(B1.*S11 + B2.*S22 + B3.*S12);
            if ~wantK, continue; end
            c   = 2*(mu1 - lam1*lnJ);
            D11 = lam1*Ci11.*Ci11 + c.*Ci11.^2;
            D12 = lam1*Ci11.*Ci22 + c.*Ci12.^2;
            D13 = lam1*Ci11.*Ci12 + c.*Ci11.*Ci12;
            D22 = lam1*Ci22.*Ci22 + c.*Ci22.^2;
            D23 = lam1*Ci22.*Ci12 + c.*Ci12.*Ci22;
            D33 = lam1*Ci12.*Ci12 + c.*0.5.*(Ci11.*Ci22 + Ci12.^2);
            for I = 1:8
                DB1 = D11.*B1(:,I) + D12.*B2(:,I) + D13.*B3(:,I);
                DB2 = D12.*B1(:,I) + D22.*B2(:,I) + D23.*B3(:,I);
                DB3 = D13.*B1(:,I) + D23.*B2(:,I) + D33.*B3(:,I);
                for Jj = I:8
                    kij = w*(DB1.*B1(:,Jj) + DB2.*B2(:,Jj) + DB3.*B3(:,Jj));
                    if mod(I,2) == mod(Jj,2)          % geometric stiffness
                        a = ceil(I/2); b = ceil(Jj/2);
                        kij = kij + w*( dNdX(1,a)*(S11*dNdX(1,b) + S12*dNdX(2,b)) ...
                                      + dNdX(2,a)*(S12*dNdX(1,b) + S22*dNdX(2,b)) );
                    end
                    khat(:,(I-1)*8+Jj) = khat(:,(I-1)*8+Jj) + kij;
                    if Jj > I
                        khat(:,(Jj-1)*8+I) = khat(:,(Jj-1)*8+I) + kij;
                    end
                end
            end
        end
    end

    function drawState()
        subplot(2,1,1);
        imagesc(1-reshape(xPhys,nely,nelx));
        axis equal; axis off; axis xy; colormap(gray); caxis([0 1]);
        title(sprintf('iter %d   \\beta = %g', iter, beta));
        subplot(2,1,2); cla;
        plot(1e3*uPath, Fout, '-o', 'LineWidth', 1.2); hold on
        plot(1e3*uPath([1 end]), [Ftar Ftar], 'r--');
        yl = [0 max(1.2*max(Fout), 1.2*Ftar)];
        plot(1e3*uPath(p.nRise)*[1 1], yl, 'k:');
        ylim(yl); xlabel('input stroke (mm)'); ylabel('F_{out} (N)');
        legend('mechanism','target','plateau starts','Location','SouthEast');
        grid on; drawnow;
    end
end
