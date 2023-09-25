def mse_loss():
    y_pred_tf = tf.matmul(Z_tf, gamma_tf)
    return loss_object(y_true=y_true_tf, y_pred=y_pred_tf)


order = 1
intercept = True 

nsp = NaturalCubicSpline(knots=knots_x, order=order, intercept=intercept)
Z = nsp.transform(knots_y, derivative=False)
dZ = nsp.transform(logtime, derivative=True)

Z_tf = tf.cast(Z, dtype=tf.float32)
y_true_tf = tf.cast(knots_y, dtype=tf.float32)

initializer = tf.keras.initializers.GlorotNormal(seed=42)
gamma = initializer(shape=(Z.shape[1], 1))
gamma_tf = tf.Variable(gamma, dtype=tf.float32)

loss_object = tf.keras.losses.MeanSquaredError()


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


optimizer_data = tf.keras.optimizers.Adam(learning_rate=1)
optimizer_spline = tf.keras.optimizers.Adam(learning_rate=1)

global_epochs = 100
local_epochs = 3

log_likelihoods, losses = [], []
for _ in range(global_epochs):
    
    # update gamma (spline)
    for _ in range(local_epochs):
        optimizer_spline.minimize(mse_loss, [gamma_tf])

    # spline matrices with current gamma estimate
    S = Z @ gamma_tf.numpy()
    dS = dZ @ gamma_tf.numpy()[1:]
    
    # update beta (likelihood)
    for _ in range(local_epochs):
        optimizer_data.minimize(neg_log_likelihood, [beta_tf])
    
    losses.append(mse_loss().numpy())
    log_likelihoods.append(np.mean(neg_log_likelihood().numpy()))