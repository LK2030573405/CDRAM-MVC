function y = rank_fun_derivative(x,delta,rank_fun)
switch rank_fun
    case 1 %Geman function
        y  = delta^2./(delta+x).^2;
    case 2 %Laplace function
        y = exp(-x./delta)/delta;
    case 3 %l_delta function
        y= (delta+delta^2)./(delta+x).^2; 
    case 4 %ETR function
        y = delta*exp(delta^2)./((x+delta).^2+eps);
    case 6 % Exponential-Geman
        inner = x./(x+delta);
        y = exp(-inner) .* delta ./ ((1-exp(-1)) .* (x+delta).^2 + eps);
    case 7 % Tanh-Geman
        inner = x./(x+delta);
        y = (1./cosh(inner).^2) .* delta ./ (tanh(1) .* (x+delta).^2 + eps);
    case 8 % Arctan-Exponential
        inner = (2/pi) .* atan(x./delta);
        y = exp(-inner) .* (2*delta) ./ (pi*(1-exp(-1)) .* (x.^2 + delta.^2) + eps);
    case 9 % Log-Tanh
        inner = log(1 + x./delta);
        y = (1./cosh(inner).^2) ./ (x + delta + eps);
    case 10 % Asinh-Exp
        inner = asinh(x./delta);
        y = exp(-inner) ./ (sqrt(x.^2 + delta.^2) + eps);
    case 11 % Asinh-Tanh
        inner = asinh(x./delta);
        y = (1./cosh(inner).^2) ./ (sqrt(x.^2 + delta.^2) + eps);
    case 12 % Laplace-Sqrt hybrid
        inner = x./delta + sqrt(x./delta);
        y = exp(-inner) .* (1./delta + 1./(2.*sqrt(delta.*x + eps)));
    case 13 % Laplace-Sqrt hybrid
        inner = x./2*delta + sqrt(x./2*delta);
        y = exp(-inner) .* (1./2*delta + 1./(2.*sqrt(2*delta.*x + eps)));
    otherwise
        error('Unsupported rank_fun id: %d', rank_fun);
end
