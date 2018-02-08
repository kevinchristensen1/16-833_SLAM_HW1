z_max = 800;
z_prior = 400;
sig = 100;
lam_short = .01;
%x = 1:1:100
normal = [];
random = [];
short = [];
for i = 1:900
    if i < z_max
        norm_exp = -0.5 .* (i - z_prior)^2/sig^2;
        normal(i) = (1./sqrt(2.*pi.*sig^2)).*exp(norm_exp);
        random(i) = 1/z_max;
    else
        normal(i) = 0;
        random(i) = 0;
    end
    if i < z_prior
        %short(i) = 1 - exp(-lam_short*150);
        short(i) = (1/(1-exp(-lam_short*z_prior))) * lam_short* exp(-lam_short*i);
    else
        short(i) = 0;
    end
    if i >= z_max
        to_far(i) = 1;
    else
        to_far(i) = 0;
    end
end
% short
dist = normal*0.7 + random*0.0988 + to_far * 0.002 + short*0.2;
% dist = short;
plot(1:900,dist);
