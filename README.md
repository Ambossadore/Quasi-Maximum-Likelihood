# Quasi-Maximum Likelihood Estimation
#### Joint Work with Jan Kallsen

This module contains the codebase for applied Quasi-Maximum Likelihood estimation of partially observed affine [2] and polynomial [1] processes. This includes:

- A class `KalmanFilter` for optimal discrete-time linear filtering of partially observed polynomial processes (see Kallsen and Richert [4])
- A base class `PolynomialModel` for calculating consistent Quasi-Maximum Likelihood estimators for a given polynomial state space model (see Kallsen and Richert [5])
- Methods for either sampling or explicitly calculating the asymptotic covariance matrix of the Quasi-Maximum Likelihood estimators
- The opportunity to instantiate any polynomial process with known infinitesimal generator as a subclass of the `PolynomialModel` class
- Two pre-implemented polynomial models (the Heston stochastic volatility model and an NIG-driven two-factor Ornstein-Uhlenbeck model)

For further details refer to Kallsen and Richert [4] as well as Kallsen and Richert [5].


### References

[1] C. Cuchiero, J. Teichmann and M. Keller-Ressel (2012). "Polynomial processes and their applications to mathematical finance". *Finance Stoch* **16**(4), pp. 711–740. https://doi.org/10.1007/s00780-012-0188-x <br>
[2] D. Duffie, D. Filipović and W. Schachermayer (2003). "Affine processes and applications in finance." *Ann. Appl. Probab.* **13**(3), pp. 984 - 1053. https://doi.org/10.1214/aoap/1060202833 <br>
[3] E. Eberlein and J. Kallsen (2019). "Mathematical Finance". Springer, Cham. <br>
[4] J. Kallsen and I. Richert (2025). "Filtering of partially observed polynomial processes in discrete and continuous time". Preprint, https://arxiv.org/abs/2503.05588. <br>
[5] J. Kallsen and I. Richert (2025). "Parameter estimation for partially observed affine and polynomial processes". Preprint, https://arxiv.org/abs/2503.05590.