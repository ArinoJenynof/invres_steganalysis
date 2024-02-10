% -------------------------------------------------------------------------
% Copyright (c) 2012 DDE Lab, Binghamton University, NY.
% All Rights Reserved.
% -------------------------------------------------------------------------
% Permission to use, copy, modify, and distribute this software for
% educational, research and non-profit purposes, without fee, and without a
% written agreement is hereby granted, provided that this copyright notice
% appears in all copies. The program is supplied "as is," without any
% accompanying services from DDE Lab. DDE Lab does not warrant the
% operation of the program will be uninterrupted or error-free. The
% end-user understands that the program was developed for research purposes
% and is advised not to rely exclusively on the program for any reason. In
% no event shall Binghamton University or DDE Lab be liable to any party
% for direct, indirect, special, incidental, or consequential damages,
% including lost profits, arising out of the use of this software. DDE Lab
% disclaims any warranties, and has no obligations to provide maintenance,
% support, updates, enhancements or modifications.
% -------------------------------------------------------------------------
% Contact: vojtech_holub@yahoo.com | fridrich@binghamton.edu | October 2012
%          http://dde.binghamton.edu/download/steganography
% -------------------------------------------------------------------------
% This function simulates embedding using WOW steganographic 
% algorithm. For more deatils about the individual submodels, please see 
% the publication [1]. 
% -------------------------------------------------------------------------
% Input:  coverPath ... path to the image
%         payload ..... payload in bits per pixel
% Output: stego ....... resulting image with embedded payload
% -------------------------------------------------------------------------
% [1] Designing Steganographic Distortion Using Directional Filters, 
% V. Holub and J. Fridrich, to be presented at WIFS'12 IEEE International 
% Workshop on Information Forensics and Security
% -------------------------------------------------------------------------
function parallel_WOW(imageCover)
  imageCover_fp = fullfile(imageCover.folder, imageCover.name);
  disp(imageCover_fp);

  [coverDir, coverName, coverExt] = fileparts(imageCover_fp);
  stegoPath = fullfile('./images_stego', strcat(coverName, '.mat'));

  stego = WOW(imageCover_fp, imageCover.payload);
  save('-mat7-binary', stegoPath, 'stego');
endfunction

function [stego, distortion] = WOW(coverPath, payload)
  hpdf = [-0.0544158422, 0.3128715909, -0.6756307363, 0.5853546837, 0.0158291053, -0.2840155430, -0.0004724846, 0.1287474266, 0.0173693010, -0.0440882539, ...
  -0.0139810279, 0.0087460940, 0.0048703530, -0.0003917404, -0.0006754494, -0.0001174768];
  lpdf = (-1).^(0:numel(hpdf)-1).*fliplr(hpdf);
  F{1} = lpdf'*hpdf;
  F{2} = hpdf'*lpdf;
  F{3} = hpdf'*hpdf;

  cover = double(imread(coverPath));
  p = -1;
  wetCost = 10^10;
  sizeCover = size(cover);

  padSize = max([size(F{1})'; size(F{2})'; size(F{3})']);
  coverPadded = padarray(cover, [padSize padSize], 'symmetric');

  xi = cell(3, 1);
  for fIndex = 1:3
    R = conv2(coverPadded, F{fIndex}, 'same');

    xi{fIndex} = conv2(abs(R), rot90(abs(F{fIndex}), 2), 'same');

    if mod(size(F{fIndex}, 1), 2) == 0, xi{fIndex} = circshift(xi{fIndex}, [1, 0]); endif;
    if mod(size(F{fIndex}, 2), 2) == 0, xi{fIndex} = circshift(xi{fIndex}, [0, 1]); endif;

    xi{fIndex} = xi{fIndex}(((size(xi{fIndex}, 1)-sizeCover(1))/2)+1:end-((size(xi{fIndex}, 1)-sizeCover(1))/2), ((size(xi{fIndex}, 2)-sizeCover(2))/2)+1:end-((size(xi{fIndex}, 2)-sizeCover(2))/2));
  endfor

  rho = ( (xi{1}.^p) + (xi{2}.^p) + (xi{3}.^p) ) .^ (-1/p);

  rho(rho > wetCost) = wetCost;
  rho(isnan(rho)) = wetCost;
  rhoP1 = rho;
  rhoM1 = rho;
  rhoP1(cover==255) = wetCost;
  rhoM1(cover==0) = wetCost;

  stego = EmbeddingSimulator(cover, rhoP1, rhoM1, payload*numel(cover), false);
  distortion_local = rho(cover~=stego);
  distortion = sum(distortion_local);
endfunction

function [y] = EmbeddingSimulator(x, rhoP1, rhoM1, m, fixEmbeddingChanges)
  n = numel(x);
  lambda = calc_lambda(rhoP1, rhoM1, m, n);
  pChangeP1 = (exp(-lambda .* rhoP1))./(1 + exp(-lambda .* rhoP1) + exp(-lambda .* rhoM1));
  pChangeM1 = (exp(-lambda .* rhoM1))./(1 + exp(-lambda .* rhoP1) + exp(-lambda .* rhoM1));
  if fixEmbeddingChanges == 1
    % RandStream.setGlobalStream(RandStream('mt19937ar','seed',139187));
    rand('state', 139187);
  else
    % RandStream.setGlobalStream(RandStream('mt19937ar','Seed',sum(100*clock)));
    rand('state', sum(100*clock));
  endif
  randChange = rand(size(x));
  y = x;
  y(randChange < pChangeP1) = y(randChange < pChangeP1) + 1;
  y(randChange >= pChangeP1 & randChange < pChangeP1+pChangeM1) = y(randChange >= pChangeP1 & randChange < pChangeP1+pChangeM1) - 1;
endfunction

function lambda = calc_lambda(rhoP1, rhoM1, message_length, n)
  l3 = 1e+3;
  m3 = double(message_length + 1);
  iterations = 0;
  while m3 > message_length
    l3 = l3 * 2;
    pP1 = (exp(-l3 .* rhoP1))./(1 + exp(-l3 .* rhoP1) + exp(-l3 .* rhoM1));
    pM1 = (exp(-l3 .* rhoM1))./(1 + exp(-l3 .* rhoP1) + exp(-l3 .* rhoM1));
    m3 = ternary_entropyf(pP1, pM1);
    iterations = iterations + 1;
    if (iterations > 10)
      lambda = l3;
      return;
    endif
  endwhile

  l1 = 0;
  m1 = double(n);
  lambda = 0;

  alpha = double(message_length)/n;
  % limit search to 30 iterations
  % and require that relative payload embedded is roughly within 1/1000 of the required relative payload
  while (double(m1-m3)/n > alpha/1000.0 ) && (iterations<30)
    lambda = l1+(l3-l1)/2;
    pP1 = (exp(-lambda .* rhoP1))./(1 + exp(-lambda .* rhoP1) + exp(-lambda .* rhoM1));
    pM1 = (exp(-lambda .* rhoM1))./(1 + exp(-lambda .* rhoP1) + exp(-lambda .* rhoM1));
    m2 = ternary_entropyf(pP1, pM1);
    if m2 < message_length
      l3 = lambda;
      m3 = m2;
    else
      l1 = lambda;
      m1 = m2;
    endif
    iterations = iterations + 1;
  endwhile
endfunction

function Ht = ternary_entropyf(pP1, pM1)
  p0 = 1-pP1-pM1;
  P = [p0(:); pP1(:); pM1(:)];
  H = -((P).*log2(P));
  H((P<eps) | (P > 1-eps)) = 0;
  Ht = sum(H);
endfunction

