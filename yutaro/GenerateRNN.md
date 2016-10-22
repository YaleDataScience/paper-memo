GenerateRNN

## Section 3: Text Prediction

Text data is discrete. If there are $$K$$ text classes in total, and class $$k$$ is fed in at time $t$,
then $x_t$ is a length $K$ vector whose entries are all zero except for the $k$ th, which is one.
Gvien an input vector (one-hot vector) $x_t \in \mathbb{R}^K$, the network models the
probability distribution of the next input $x_{t+1}$, given $y_t$. That is,

$$y_t = RNN(x_t) \ x_{t+1} | y_t\sim multinomial(y_t) \ \iff P(x_{t+1} = k | y_t) = y_t^k = \frac{ \exp(\hat{y_t}^k)}{\sum_i \exp(\hat{y_t}^i) }$$

$$$$

