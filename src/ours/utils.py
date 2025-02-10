import numpy as np
from sklearn.preprocessing import PolynomialFeatures


def create_polynomial_representation(
    X, degree, use_sklearn: bool = False, interaction_only: bool = False
):
    # if degree <= 1:
    #    raise ValueError("Degree must be larger than 2.")

    if interaction_only and not use_sklearn:
        print("Warning: interaction_only has no effect as use_sklearn = False.")

    if use_sklearn:
        return PolynomialFeatures(
            degree=degree, interaction_only=interaction_only, include_bias=False
        ).fit_transform(X)

    else:
        n_features = X.shape[1]

        # Create an empty list to store polynomial features
        poly_features = []

        # Iterate over each feature
        for feature_idx in range(n_features):
            # Create polynomial features for the current feature
            feature = X[:, feature_idx]
            poly_feature = np.column_stack([feature**d for d in range(1, degree + 1)])
            poly_features.append(poly_feature)

        # Stack the polynomial features horizontally
        X_poly = np.hstack(poly_features)
        return X_poly
