# heart attack survival data
X, y = load_whas500()
delta, time = list(zip(*y))

# retain only numerical covariates 
X = X.loc[:, ["age", "bmi", "diasbp", "hr", "los", "sysbp"]].values

scaler = StandardScaler()
X = scaler.fit_transform(X)

delta = np.array(outcome)
logtime = np.log(np.array(time))

# HACK
to_keep = logtime > 0

y = y[to_keep]
X = X[to_keep]
delta = delta[to_keep]
logtime = logtime[to_keep]


# knot locations are at the centiles of the distribution of *uncensored* log event times
knots_x = np.linspace(0, 1, 6)
knots_y = np.ones(len(knots_x)) * float(np.nan)

# uncensored event times 
logtime_uncens = logtime[delta == 0]

# interior points
for i in range(1, len(knots_x) - 1):
    knots_y[i] = np.quantile(logtime_uncens, knots_x[i])

# boundaries
knots_y[0] = min(logtime_uncens)
knots_y[-1] = max(logtime_uncens)

# NOTE: always monotonic (rel. to ln(dS) in data likelihood)
plt.figure()
plt.plot(knots_x, knots_y, marker="o", linestyle="")
plt.plot(np.linspace(0, 1, len(logtime_uncens)), sorted(logtime_uncens))


# linear regression model with transformed covariates (features)
def mse_loss():
    y_pred_tf = tf.matmul(Z_tf, gamma_tf)
    return loss_object(y_true=y_true_tf, y_pred=y_pred_tf)


order = 1
intercept = True 

nsp = NaturalCubicSpline(knots=knots_x, order=order, intercept=intercept)
Z = nsp.transform(knots_y, derivative=False)
Z_tf = tf.cast(Z, dtype=tf.float32)

y_true_tf = tf.cast(knots_y, dtype=tf.float32)

initializer = tf.keras.initializers.GlorotNormal(seed=42)
gamma = initializer(shape=(Z.shape[1], 1))
gamma_tf = tf.Variable(gamma, dtype=tf.float32)

optimizer = tf.keras.optimizers.Adam(learning_rate=1)
loss_object = tf.keras.losses.MeanSquaredError()

epochs = 200

losses = []
for epoch in range(epochs):
    
    optimizer.minimize(mse_loss, [gamma_tf])
    losses.append(mse_loss())


gamma_star = gamma_tf.numpy()
knots_y_hat = (Z @ gamma_star).squeeze()

plt.figure()
plt.plot(knots_x, knots_y, marker="o", linestyle="", label="actual")
plt.plot(knots_x, knots_y_hat, marker="o", linestyle="", label="GD")
plt.legend()

print("avg error:", np.mean(knots_y - knots_y_hat))



# re-use results from spline optimisation 
nsp = NaturalCubicSpline(knots=knots_x, order=order, intercept=intercept)
Z = nsp.transform(logtime, derivative=False)
dZ = nsp.transform(logtime, derivative=True)

# spline matrices 
S = Z @ gamma_star
# first term vanishes in derivative 
dS = dZ @ gamma_star[1:]


def neg_log_likelihood():
    
    nu = S_tf + tf.matmul(X_tf, beta_tf)
    log_likelihood = delta_tf * (tf.math.log(dS_tf) + nu) - tf.exp(nu)

    return -1.0 * log_likelihood


initializer = tf.keras.initializers.GlorotNormal(seed=42)
beta = initializer(shape=(X.shape[1], 1), dtype=tf.float32)
beta_tf = tf.Variable(beta, dtype=tf.float32)

X_tf = tf.cast(X, dtype=tf.float32)
S_tf = tf.cast(S, dtype=tf.float32)
dS_tf = tf.cast(dS, dtype=tf.float32)
delta_tf = tf.cast(delta, dtype=tf.float32)
optimizer = tf.keras.optimizers.Adam(learning_rate=1)

epochs = 200

log_likelihoods = []
for epoch in range(epochs):
    
    optimizer.minimize(neg_log_likelihood, [beta_tf])
    log_likelihoods.append(np.mean(neg_log_likelihood().numpy()))


y_pred = ((X @ beta.numpy()).squeeze() > 0).astype(int)
confusion_matrix(delta, y_pred)



# baseline CoxPHSurvivalAnalysis
model = CoxPHSurvivalAnalysis()
model.fit(X, y)

y_pred = (model.predict(X).squeeze() > 0).astype(int)

confusion_matrix(delta, y_pred)