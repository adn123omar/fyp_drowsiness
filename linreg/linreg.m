function [a0,a1,r2] = linreg(x,y)
% [a0,a1,r2] = linreg(x, y) 
% Written by: ???, ID: ???
% Last modified: ???
% Performs linear regression on the linear x and y data set 
%
% INPUTS:
%  - x: linear independent data set 
%  - y: linear dependent data set 
% OUTPUT:
%  - a0: constant in y=a1*x + a0 
%  - a1: gradient in y=a1*x + a0
%  - r2: coefficient of determination

% Getting best regression coefficients
n = length(x);
Sx = sum(x);
Sy = sum(y);
Sxx = sum(x.*x);
Sxy = sum(x.*y);
a1 = (n*Sxy - Sx*Sy)/(n*Sxx - Sx^2);
a0 = mean(y) - a1*mean(x);

% Getting r2 value
St = sum((y - mean(y)).^2);
Sr = sum((y - a0 - a1*x).^2);
r2 = (St - Sr)/St;
