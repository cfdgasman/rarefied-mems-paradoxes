"""Published reference values for validation.

Sharipov & Seleznev, "Data on internal rarefied gas flows", J. Phys. Chem. Ref.
Data 27 (1998) 657, Table 1: reduced flow rate G_P of plane Poiseuille flow,
linearised BGK model, diffuse walls.  Columns:
    a  Cercignani & Daneri (1963), direct numerical method
    b  Cercignani & Pagani (1966), variational method
    c  Huang, Hwang & Wang (1995), discrete velocity method
"""

SHARIPOV_TABLE1: dict[float, tuple[float, float, float]] = {
    0.01: (3.0499, 3.0489, 2.2114),
    0.1: (2.0328, 2.0314, 1.9829),
    0.2: (1.8083, 1.8079, 1.8167),
    0.5: (1.6017, 1.6017, 1.6050),
    1.0: (1.5379, 1.5389, 1.5381),
    2.0: (1.5912, 1.5942, 1.5950),
    4.0: (1.8450, 1.8440, 1.8459),
    5.0: (1.9895, 1.9883, 1.9908),
    7.0: (2.2904, 2.2914, 2.2945),
    10.0: (2.7558, 2.7638, 2.7681),
}
