function result = top3d(s)
%TOP3D 3-D compliant-mechanism topology-optimization solver.
%
% This project version follows the Top3d structured finite-element and SIMP
% approach, with the two-load-case compliant-mechanism formulation.
%
% Input:
%   s - settings structure created in the Live Script
%
% Output:
%   result.xPhys                  optimized density field
%   result.history                convergence history
%   result.unitInputDisplacement  input displacement under unit load
%   result.unitOutputDisplacement output displacement under unit load
%   result.inputStiffness         normalized input stiffness

nelx = s.nelx;
nely = s.nely;
nelz = s.nelz;

nele = nelx*nely*nelz;
ndof = 3*(nelx+1)*(nely+1)*(nelz+1);

inputDOF = s.BC.inputDOF;
outputDOF = s.BC.outputDOF;
fixedDOFs = s.BC.fixedDOFs;

F = sparse([inputDOF outputDOF],[1 2],[-1 -1],ndof,2);
U = zeros(ndof,2);

freeDOFs = setdiff((1:ndof)',fixedDOFs);

KE = elementStiffnessH8(s.nu);

nodeGrid = reshape(1:(nely+1)*(nelx+1),nely+1,nelx+1);
nodeIDs = reshape(nodeGrid(1:end-1,1:end-1),nely*nelx,1);

nodeIDz = 0:(nely+1)*(nelx+1):(nelz-1)*(nely+1)*(nelx+1);
nodeIDs = repmat(nodeIDs,size(nodeIDz)) + ...
          repmat(nodeIDz,size(nodeIDs));

edofVec = 3*nodeIDs(:)+1;

edofMat = repmat(edofVec,1,24) + repmat( ...
    [0 1 2, ...
     3*nely+[3 4 5 0 1 2], ...
     -3 -2 -1, ...
     3*(nely+1)*(nelx+1)+[0 1 2, ...
     3*nely+[3 4 5 0 1 2], ...
     -3 -2 -1]],nele,1);

iK = reshape(kron(edofMat,ones(24,1))',24*24*nele,1);
jK = reshape(kron(edofMat,ones(1,24))',24*24*nele,1);

[H,Hs] = densityFilter(nelx,nely,nelz,s.rmin);

x = s.volfrac*ones(nely,nelx,nelz);
x(s.design.passiveSolid) = 1;
x(s.design.passiveVoid) = 0;

xPhys = reshape((H*x(:))./Hs,size(x));
xPhys(s.design.passiveSolid) = 1;
xPhys(s.design.passiveVoid) = 0;

history.iteration = zeros(s.maxIter,1);
history.objective = zeros(s.maxIter,1);
history.volume = zeros(s.maxIter,1);
history.change = zeros(s.maxIter,1);

if ~isfield(s,"animation")
    s.animation.show = true;
    s.animation.store = true;
    s.animation.threshold = 0.50;
    s.animation.every = 1;
end

if ~isfield(s.animation,"show"),      s.animation.show = true; end
if ~isfield(s.animation,"store"),     s.animation.store = true; end
if ~isfield(s.animation,"threshold"), s.animation.threshold = 0.50; end
if ~isfield(s.animation,"every"),     s.animation.every = 1; end

if s.animation.store
    xHistory = cell(s.maxIter,1);
else
    xHistory = {};
end

if s.animation.show
    animationFigure = figure("Name","Topology Optimization Progress");
else
    animationFigure = [];
end

iteration = 0;
change = 1;

while change > s.tolX && iteration < s.maxIter

    iteration = iteration+1;

    modulus = s.Emin + ...
        xPhys(:)'.^s.penal*(s.E0-s.Emin);

    sK = reshape(KE(:)*modulus,24*24*nele,1);

    K = sparse(iK,jK,sK,ndof,ndof);
    K = (K+K')/2;

    % Stabilizing springs used for compliant-mechanism synthesis.
    springStiffness = 0.10;

    K(inputDOF,inputDOF) = ...
        K(inputDOF,inputDOF)+springStiffness;

    K(outputDOF,outputDOF) = ...
        K(outputDOF,outputDOF)+springStiffness;

    U(:,:) = 0;
    U(freeDOFs,:) = K(freeDOFs,freeDOFs)\F(freeDOFs,:);

    U1 = U(:,1);
    U2 = U(:,2);

    ce = reshape(sum((U1(edofMat)*KE).*U2(edofMat),2), ...
        [nely,nelx,nelz]);

    objective = U(outputDOF,1);

    dc = s.penal*(s.E0-s.Emin).* ...
         xPhys.^(s.penal-1).*ce;

    dv = ones(nely,nelx,nelz);

    dc(:) = H*(dc(:)./Hs);
    dv(:) = H*(dv(:)./Hs);

    locked = s.design.passiveSolid | s.design.passiveVoid;
    dc(locked) = 0;
    dv(locked) = 0;

    lower = 0;
    upper = 1e9;
    move = 0.10;

    while (upper-lower)/(upper+lower+eps) > 1e-4

        middle = 0.5*(upper+lower);

        updateFactor = ...
            max(1e-10,-dc./max(dv,eps)/middle).^0.30;

        xNew = max(0,max(x-move, ...
            min(1,min(x+move,x.*updateFactor))));

        xNew(s.design.passiveSolid) = 1;
        xNew(s.design.passiveVoid) = 0;

        trialPhys = reshape((H*xNew(:))./Hs,size(xNew));
        trialPhys(s.design.passiveSolid) = 1;
        trialPhys(s.design.passiveVoid) = 0;

        if mean(trialPhys(:)) > s.volfrac
            lower = middle;
        else
            upper = middle;
        end
    end

    change = max(abs(xNew(:)-x(:)));

    x = xNew;
    xPhys = trialPhys;

    history.iteration(iteration) = iteration;
    history.objective(iteration) = objective;
    history.volume(iteration) = mean(xPhys(:));
    history.change(iteration) = change;

    fprintf("Iteration %3d | objective % .5e | volume %.3f | change %.3f\n", ...
        iteration,objective,mean(xPhys(:)),change);

    if s.animation.store
        xHistory{iteration} = xPhys;
    end

    if s.animation.show && mod(iteration-1,s.animation.every)==0
        helper("updateAnimation",animationFigure,xPhys, ...
            s.animation.threshold,iteration);
    end
end

modulus = s.Emin + xPhys(:)'.^s.penal*(s.E0-s.Emin);
sK = reshape(KE(:)*modulus,24*24*nele,1);

K = sparse(iK,jK,sK,ndof,ndof);
K = (K+K')/2;

springStiffness = 0.10;
K(inputDOF,inputDOF) = K(inputDOF,inputDOF)+springStiffness;
K(outputDOF,outputDOF) = K(outputDOF,outputDOF)+springStiffness;

U(:,:) = 0;
U(freeDOFs,:) = K(freeDOFs,freeDOFs)\F(freeDOFs,:);

unitInputDisplacement = U(inputDOF,1);
unitOutputDisplacement = U(outputDOF,1);

result.x = x;
result.xPhys = xPhys;
result.U = U;
result.iterations = iteration;
result.unitInputDisplacement = unitInputDisplacement;
result.unitOutputDisplacement = unitOutputDisplacement;
result.inputStiffness = 1/max(abs(unitInputDisplacement),eps);

result.history.iteration = history.iteration(1:iteration);
result.history.objective = history.objective(1:iteration);
result.history.volume = history.volume(1:iteration);
result.history.change = history.change(1:iteration);

if s.animation.store
    result.xHistory = xHistory(1:iteration);
else
    result.xHistory = {};
end
end

function [H,Hs] = densityFilter(nelx,nely,nelz,rmin)

nele = nelx*nely*nelz;
radius = ceil(rmin)-1;
maximumEntries = nele*(2*radius+1)^3;

iH = zeros(maximumEntries,1);
jH = zeros(maximumEntries,1);
sH = zeros(maximumEntries,1);

counter = 0;

for k1 = 1:nelz
    for i1 = 1:nelx
        for j1 = 1:nely

            e1 = (k1-1)*nelx*nely+(i1-1)*nely+j1;

            for k2 = max(k1-radius,1):min(k1+radius,nelz)
                for i2 = max(i1-radius,1):min(i1+radius,nelx)
                    for j2 = max(j1-radius,1):min(j1+radius,nely)

                        e2 = (k2-1)*nelx*nely+(i2-1)*nely+j2;

                        weight = rmin-sqrt( ...
                            (i1-i2)^2+(j1-j2)^2+(k1-k2)^2);

                        if weight > 0
                            counter = counter+1;
                            iH(counter) = e1;
                            jH(counter) = e2;
                            sH(counter) = weight;
                        end
                    end
                end
            end
        end
    end
end

H = sparse(iH(1:counter),jH(1:counter), ...
           sH(1:counter),nele,nele);

Hs = sum(H,2);
end

function KE = elementStiffnessH8(nu)

A = [32 6 -8 6 -6 4 3 -6 -10 3 -3 -3 -4 -8;
    -48 0 0 -24 24 0 0 0 12 -12 0 12 12 12];

k = (1/144)*A'*[1;nu];

K1 = [k(1) k(2) k(2) k(3) k(5) k(5);
      k(2) k(1) k(2) k(4) k(6) k(7);
      k(2) k(2) k(1) k(4) k(7) k(6);
      k(3) k(4) k(4) k(1) k(8) k(8);
      k(5) k(6) k(7) k(8) k(1) k(2);
      k(5) k(7) k(6) k(8) k(2) k(1)];

K2 = [k(9) k(8) k(12) k(6) k(4) k(7);
      k(8) k(9) k(12) k(5) k(3) k(5);
      k(10) k(10) k(13) k(7) k(4) k(6);
      k(6) k(5) k(11) k(9) k(2) k(10);
      k(4) k(3) k(5) k(2) k(9) k(12);
      k(7) k(5) k(4) k(10) k(12) k(13)];

K3 = [k(6) k(7) k(4) k(9) k(12) k(8);
      k(7) k(6) k(4) k(10) k(13) k(10);
      k(5) k(5) k(3) k(8) k(12) k(9);
      k(9) k(10) k(2) k(6) k(11) k(5);
      k(12) k(13) k(10) k(11) k(6) k(4);
      k(2) k(12) k(9) k(4) k(5) k(3)];

K4 = [k(14) k(11) k(11) k(13) k(10) k(10);
      k(11) k(14) k(11) k(12) k(9) k(8);
      k(11) k(11) k(14) k(12) k(8) k(9);
      k(13) k(12) k(12) k(14) k(7) k(7);
      k(10) k(9) k(8) k(7) k(14) k(11);
      k(10) k(8) k(9) k(7) k(11) k(14)];

K5 = [k(1) k(2) k(8) k(3) k(5) k(4);
      k(2) k(1) k(8) k(4) k(6) k(11);
      k(8) k(8) k(1) k(5) k(11) k(6);
      k(3) k(4) k(5) k(1) k(8) k(2);
      k(5) k(6) k(11) k(8) k(1) k(8);
      k(4) k(11) k(6) k(2) k(8) k(1)];

K6 = [k(14) k(11) k(7) k(13) k(10) k(12);
      k(11) k(14) k(7) k(12) k(9) k(2);
      k(7) k(7) k(14) k(10) k(2) k(9);
      k(13) k(12) k(10) k(14) k(7) k(11);
      k(10) k(9) k(2) k(7) k(14) k(7);
      k(12) k(2) k(9) k(11) k(7) k(14)];

KE = 1/((nu+1)*(1-2*nu))* ...
    [K1 K2 K3 K4;
     K2' K5 K6 K3';
     K3' K6 K5' K2';
     K4 K3 K2 K1'];
end
