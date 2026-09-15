function varargout = helper(action,varargin)
%HELPER Supporting functions for the compliant-grasper Live Script.
%
% Examples:
%   design = helper("createDesign",model);
%   BC = helper("createBC",model,design);
%   helper("showTopology",rho,0.5);

action = lower(string(action));

switch action
    case "createdesign"
        varargout{1} = createDesign(varargin{1});

    case "showdesign"
        showDesign(varargin{1});

    case "createbc"
        varargout{1} = createBC(varargin{1},varargin{2});

    case "showbc"
        showBC(varargin{1},varargin{2},varargin{3});

    case "showtopology"
        showTopology(varargin{1},varargin{2});

    case "updateanimation"
        updateAnimation(varargin{1},varargin{2},varargin{3},varargin{4});

    case "playanimation"
        playAnimation(varargin{1},varargin{2},varargin{3});

    case "savegif"
        saveGIF(varargin{1},varargin{2},varargin{3},varargin{4});

    case "plothistory"
        plotHistory(varargin{1});

    case "forceresponse"
        varargout{1} = forceResponse(varargin{1},varargin{2});

    case "plotforceresponse"
        plotForceResponse(varargin{1});

    case "exportstl"
        exportSTL(varargin{1},varargin{2},varargin{3},varargin{4});

    case "nodenumber"
        varargout{1} = nodeNumber(varargin{:});

    otherwise
        error("helper:UnknownAction","Unknown helper action: %s",action);
end
end

function design = createDesign(model)

nely = model.nely;
nelx = model.nelx;
nelz = model.nelz;

passiveSolid = false(nely,nelx,nelz);
passiveVoid = false(nely,nelx,nelz);

baseLength = 4;
padLength = 4;

% Fixed mounting block.
passiveSolid(:,1:baseLength,:) = true;

% Actuator pad.
inputRows = 2:6;
inputCols = baseLength+1:baseLength+padLength;
passiveSolid(inputRows,inputCols,:) = true;

% Distal tissue-contact pad.
jawRows = round(0.40*nely):round(0.65*nely);
jawCols = nelx-padLength+1:nelx;
passiveSolid(jawRows,jawCols,:) = true;

% Clearance region.
voidRows = round(0.70*nely):nely;
voidCols = round(0.35*nelx):nelx;
passiveVoid(voidRows,voidCols,:) = true;

passiveVoid(passiveSolid) = false;

design.passiveSolid = passiveSolid;
design.passiveVoid = passiveVoid;
design.baseLength = baseLength;
design.padLength = padLength;
design.inputRows = inputRows;
design.inputCols = inputCols;
design.jawRows = jawRows;
design.jawCols = jawCols;
end

function showDesign(design)

region = 0.35*ones(size(design.passiveSolid));
region(design.passiveSolid) = 1;
region(design.passiveVoid) = 0;

figure("Name","Initial design domain");
displayDensity3D(region,0.10);
title("Initial grasper design domain");
end

function BC = createBC(model,design)

midThickness = round(model.nelz/2);

input_i = design.baseLength+design.padLength;
input_j = 3;
input_k = midThickness;

output_i = model.nelx;
output_j = round(0.52*model.nely);
output_k = midThickness;

inputNode = nodeNumber(input_i,input_j,input_k, ...
    model.nelx,model.nely);

outputNode = nodeNumber(output_i,output_j,output_k, ...
    model.nelx,model.nely);

BC.inputNode = inputNode;
BC.outputNode = outputNode;
BC.inputDOF = 3*inputNode-1;
BC.outputDOF = 3*outputNode-1;

[jGrid,kGrid] = meshgrid(0:model.nely,0:model.nelz);

fixedNodes = nodeNumber(zeros(size(jGrid)),jGrid,kGrid, ...
    model.nelx,model.nely);

BC.fixedNodes = unique(fixedNodes(:));
BC.fixedDOFs = unique([3*BC.fixedNodes-2;
                       3*BC.fixedNodes-1;
                       3*BC.fixedNodes]);
end

function showBC(model,design,BC)

figure("Name","Boundary conditions");
region = 0.25*ones(size(design.passiveSolid));
region(design.passiveSolid) = 0.75;
region(design.passiveVoid) = 0;

displayDensity3D(region,0.10);
hold on;

[inputX,inputY,inputZ] = nodeCoordinates( ...
    BC.inputNode,model.nelx,model.nely);

[outputX,outputY,outputZ] = nodeCoordinates( ...
    BC.outputNode,model.nelx,model.nely);

scatter3(inputX,inputY,inputZ,90,"filled");
scatter3(outputX,outputY,outputZ,90,"filled");

text(inputX,inputY,inputZ,"  Input");
text(outputX,outputY,outputZ,"  Output");

title("Input, output and fixed mounting region");
hold off;
end

function showTopology(rho,threshold)

figure("Name","Optimized compliant jaw");
displayDensity3D(rho,threshold);
title("Topology-optimized compliant jaw");
end

function plotHistory(result)

figure("Name","Optimization objective");
plot(result.history.iteration,result.history.objective, ...
    "-o","LineWidth",1.2,"MarkerSize",3);
xlabel("Iteration");
ylabel("Output-displacement objective");
title("Optimization history");
grid on;

figure("Name","Material fraction");
plot(result.history.iteration,result.history.volume, ...
    "-o","LineWidth",1.2,"MarkerSize",3);
xlabel("Iteration");
ylabel("Material fraction");
title("Material-volume history");
grid on;
end

function response = forceResponse(result,displacement)

response.displacement = displacement;
response.force = result.inputStiffness*displacement;
response.normalizedStiffness = result.inputStiffness;
response.motionRatio = result.unitOutputDisplacement / ...
    max(abs(result.unitInputDisplacement),eps);

fprintf("Normalized input stiffness: %.6f\n", ...
    response.normalizedStiffness);
fprintf("Output/input displacement ratio: %.6f\n", ...
    response.motionRatio);
end

function plotForceResponse(response)

figure("Name","Force-displacement response");
plot(response.displacement,response.force, ...
    "o-","LineWidth",1.4);
xlabel("Input displacement, mm");
ylabel("Normalized reaction force");
title("Initial linear force-displacement response");
grid on;
end

function exportSTL(rho,threshold,elementSize,fileName)

binaryVolume = rho >= threshold;
binaryVolume = localPad(binaryVolume,[1 1 1],false);
smoothVolume = smooth3(double(binaryVolume),"box",3);

surfaceData = isosurface(smoothVolume,0.50);

if isempty(surfaceData.faces)
    error("helper:STLExport","No STL surface was generated.");
end

vertices = (surfaceData.vertices-1)*elementSize;
TR = triangulation(surfaceData.faces,vertices);

stlwrite(TR,fileName);
end

function nid = nodeNumber(i,j,k,nelx,nely)

nid = k.*(nelx+1).*(nely+1) + ...
      i.*(nely+1) + ...
      (nely+1-j);
end

function [x,y,z] = nodeCoordinates(nid,nelx,nely)

zeroID = nid-1;
layerSize = (nelx+1)*(nely+1);

z = floor(zeroID/layerSize);
remainder = zeroID-z*layerSize;

x = floor(remainder/(nely+1));
localIndex = remainder-x*(nely+1);

y = nely-localIndex;
end

function displayDensity3D(rho,threshold)

[nely,nelx,nelz] = size(rho);

faces = [1 2 3 4;
         2 6 7 3;
         4 3 7 8;
         1 5 8 4;
         1 2 6 5;
         5 6 7 8];

hold on;

for k = 1:nelz
    for i = 1:nelx
        for j = 1:nely
            if rho(j,i,k) > threshold

                x = i-1;
                y = nely-j;
                z = k-1;

                vertices = [x   y   z;
                            x   y+1 z;
                            x+1 y+1 z;
                            x+1 y   z;
                            x   y   z+1;
                            x   y+1 z+1;
                            x+1 y+1 z+1;
                            x+1 y   z+1];

                shade = 0.20+0.80*(1-rho(j,i,k));

                patch("Faces",faces, ...
                      "Vertices",vertices, ...
                      "FaceColor",[shade shade shade], ...
                      "EdgeColor","none");
            end
        end
    end
end

axis equal;
axis tight;
axis vis3d;
xlabel("Length");
ylabel("Height");
zlabel("Thickness");
view(35,25);
camlight;
lighting gouraud;
grid on;
box on;
hold off;
end


function updateAnimation(fig,rho,threshold,iteration)
if isempty(fig) || ~isvalid(fig)
    return;
end
figure(fig);
clf(fig);
displayDensity3D(rho,threshold);
title(sprintf("Topology Optimization - Iteration %d",iteration));
drawnow;
end

function playAnimation(xHistory,threshold,pauseTime)
if nargin < 3
    pauseTime = 0.10;
end
fig = figure("Name","Topology Optimization Animation");
for iteration = 1:numel(xHistory)
    clf(fig);
    displayDensity3D(xHistory{iteration},threshold);
    title(sprintf("Topology Optimization - Iteration %d / %d", ...
        iteration,numel(xHistory)));
    drawnow;
    pause(pauseTime);
end
end

function saveGIF(xHistory,threshold,fileName,delayTime)
if nargin < 4
    delayTime = 0.10;
end
fig = figure("Name","GIF Generation","Visible","off");
for iteration = 1:numel(xHistory)
    clf(fig);
    displayDensity3D(xHistory{iteration},threshold);
    title(sprintf("Topology Optimization - Iteration %d / %d", ...
        iteration,numel(xHistory)));
    drawnow;
    frame = getframe(fig);
    imageRGB = frame2im(frame);
    [imageIndexed,map] = rgb2ind(imageRGB,256);
    if iteration == 1
        imwrite(imageIndexed,map,fileName,"gif", ...
            "LoopCount",inf,"DelayTime",delayTime);
    else
        imwrite(imageIndexed,map,fileName,"gif", ...
            "WriteMode","append","DelayTime",delayTime);
    end
end
close(fig);
fprintf("Animation saved: %s\n",fileName);
end

function padded = localPad(array,padSize,padValue)

newSize = size(array)+2*padSize;
padded = repmat(padValue,newSize);

rows = (1:size(array,1))+padSize(1);
cols = (1:size(array,2))+padSize(2);
deps = (1:size(array,3))+padSize(3);

padded(rows,cols,deps) = array;
end
