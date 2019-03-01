function X = logsig_output_to_derivative(P)
    X = P.*(1-P);
end