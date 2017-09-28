"""Deep Q learning graph

The functions in this file can are used to create the following functions:

======= act ========

    Function to chose an action given an observation

    Parameters
    ----------
    observation: object
        Observation that can be feed into the output of make_obs_ph
    stochastic: bool
        if set to False all the actions are always deterministic (default False)
    update_eps_ph: float
        update epsilon a new value, if negative not update happens
        (default: no update)

    Returns
    -------
    Tensor of dtype tf.int64 and shape (BATCH_SIZE,) with an action to be performed for
    every element of the batch.

======= state =======
    TBA

======= train =======

    Function that takes a transition (s,a,r,s') and optimizes Bellman equation's error:

        td_error = Q(s,a) - (r + gamma * max_a' Q(s', a'))
        loss = hauber_loss[td_error]

    Parameters
    ----------
    obs_t: object
        a batch of observations
    action: np.array
        actions that were selected upon seeing obs_t.
        dtype must be int32 and shape must be (batch_size,)
    reward: np.array
        immediate reward attained after executing those actions
        dtype must be float32 and shape must be (batch_size,)
    obs_tp1: object
        observations that followed obs_t
    done: np.array
        1 if obs_t was the last observation in the episode and 0 otherwise
        obs_tp1 gets ignored, but must be of the valid shape.
        dtype must be float32 and shape must be (batch_size,)
    weight: np.array
        imporance weights for every element of the batch (gradient is multiplied
        by the importance weight) dtype must be float32 and shape must be (batch_size,)

    Returns
    -------
    td_error: np.array
        a list of differences between Q(s,a) and the target in Bellman's equation.
        dtype is float32 and shape is (batch_size,)

======= update_target ========

    copy the parameters from optimized Q function to the target Q function.
    In Q learning we actually optimize the following error:

        Q(s,a) - (r + gamma * max_a' Q'(s', a'))

    Where Q' is lagging behind Q to stablize the learning. For example for Atari

    Q' is set to Q once every 10000 updates training steps.

"""
import tensorflow as tf
import baselines.common.tf_util as U
import os


def build_aux(make_obs_ph, aux_func, num_actions, num_outputs, scope="deepq", name_aux='aux', reuse=None, dueling=True):
    """Introduction TBA
        Note that scope_aux is for checking if the optimization procedure is doing the right thing!
    """
    with tf.variable_scope(scope, reuse=reuse):
        observations_ph = U.ensure_tf_input(make_obs_ph("observation"))
        act_t_ph = tf.placeholder(tf.int32, [None], name='action')
        output_aux = aux_func(observations_ph.get(), act_t_ph, num_actions, num_outputs,
                              scope=name_aux+'_func',
                              name_aux=name_aux, dueling=dueling)
        aux = U.function(inputs=[observations_ph, act_t_ph], outputs=[output_aux])

    return aux


def build_act_state(make_obs_ph, q_func, num_actions, scope="deepq", reuse=None):
    """Creates the act function:

    Parameters
    ----------
    make_obs_ph: str -> tf.placeholder or TfInput
        a function that take a name and creates a placeholder of input with that name
    q_func: (tf.Variable, int, str, bool) -> tf.Variable
        the model that takes the following inputs:
            observation_in: object
                the output of observation placeholder
            num_actions: int
                number of actions
            scope: str
            reuse: bool
                should be passed to outer variable scope
        and returns a tensor of shape (batch_size, num_actions) with values of every action.
    num_actions: int
        number of actions.
    scope: str or VariableScope
        optional scope for variable_scope.
    reuse: bool or None
        whether or not the variables should be reused. To be able to reuse the scope must be given.

    Returns
    -------
    act: (tf.Variable, bool, float) -> tf.Variable
        function to select and action given observation.
`       See the top of the file for details.
    """
    with tf.variable_scope(scope, reuse=reuse):
        observations_ph = U.ensure_tf_input(make_obs_ph("observation"))
        stochastic_ph = tf.placeholder(tf.bool, (), name="stochastic")
        update_eps_ph = tf.placeholder(tf.float32, (), name="update_eps")

        eps = tf.get_variable("eps", (), initializer=tf.constant_initializer(0))

        q_values, state = q_func(observations_ph.get(), num_actions, scope="q_func")

        deterministic_actions = tf.argmax(q_values, axis=1)

        batch_size = tf.shape(observations_ph.get())[0]
        random_actions = tf.random_uniform(tf.stack([batch_size]), minval=0, maxval=num_actions, dtype=tf.int64)
        chose_random = tf.random_uniform(tf.stack([batch_size]), minval=0, maxval=1, dtype=tf.float32) < eps
        stochastic_actions = tf.where(chose_random, random_actions, deterministic_actions)

        output_actions = tf.cond(stochastic_ph, lambda: stochastic_actions, lambda: deterministic_actions)
        update_eps_expr = eps.assign(tf.cond(update_eps_ph >= 0, lambda: update_eps_ph, lambda: eps))

        act = U.function(inputs=[observations_ph, stochastic_ph, update_eps_ph],
                         outputs=output_actions,
                         givens={update_eps_ph: -1.0, stochastic_ph: True},
                         updates=[update_eps_expr])

        state_f = U.function(inputs=[observations_ph],
                             outputs=state)

        return act, state_f


def build_predictors_joint(make_obs_ph,
                          q_cnn, q_postcnn, aux_postcnn,
                          num_actions, dim_state=512, scope="deepq", reuse=None):
    """Build act for joint model version"""
    with tf.variable_scope(scope, reuse=reuse):
        observations_ph = U.ensure_tf_input(make_obs_ph("observation"))
        stochastic_ph = tf.placeholder(tf.bool, (), name="stochastic")
        update_eps_ph = tf.placeholder(tf.float32, (), name="update_eps")
        act_t_ph = tf.placeholder(tf.int32, [None], name="action")

        eps = tf.get_variable("eps", (), initializer=tf.constant_initializer(0))

        # q_values, state = q_func(observations_ph.get(), num_actions, scope="q_func")
        with tf.variable_scope("q_func", reuse=reuse):
            state = q_cnn(observations_ph.get(), dim_state=dim_state, scope="cnn_feature", reuse=False)
            q_values = q_postcnn(state, num_actions, scope="q_func_post_cnn", reuse=False)
            sh_tp1, rh_t = aux_postcnn(state, act_t_ph, num_actions, scope="aux_func_post_cnn", reuse=False)

        deterministic_actions = tf.argmax(q_values, axis=1)

        batch_size = tf.shape(observations_ph.get())[0]
        random_actions = tf.random_uniform(tf.stack([batch_size]), minval=0, maxval=num_actions, dtype=tf.int64)
        chose_random = tf.random_uniform(tf.stack([batch_size]), minval=0, maxval=1, dtype=tf.float32) < eps
        stochastic_actions = tf.where(chose_random, random_actions, deterministic_actions)

        output_actions = tf.cond(stochastic_ph, lambda: stochastic_actions, lambda: deterministic_actions)
        update_eps_expr = eps.assign(tf.cond(update_eps_ph >= 0, lambda: update_eps_ph, lambda: eps))

        act = U.function(inputs=[observations_ph, stochastic_ph, update_eps_ph],
                         outputs=output_actions,
                         givens={update_eps_ph: -1.0, stochastic_ph: True},
                         updates=[update_eps_expr])

        state_f = U.function(inputs=[observations_ph],
                             outputs=state)

        spred = U.function(inputs=[observations_ph, act_t_ph],
                           outputs=sh_tp1)

        rpred = U.function(inputs=[observations_ph, act_t_ph],
                           outputs=rh_t)

        return act, state_f, spred, rpred


def build_act(make_obs_ph, q_func, num_actions, scope="deepq", reuse=None):
    """Creates the act function:

    Parameters
    ----------
    make_obs_ph: str -> tf.placeholder or TfInput
        a function that take a name and creates a placeholder of input with that name
    q_func: (tf.Variable, int, str, bool) -> tf.Variable
        the model that takes the following inputs:
            observation_in: object
                the output of observation placeholder
            num_actions: int
                number of actions
            scope: str
            reuse: bool
                should be passed to outer variable scope
        and returns a tensor of shape (batch_size, num_actions) with values of every action.
    num_actions: int
        number of actions.
    scope: str or VariableScope
        optional scope for variable_scope.
    reuse: bool or None
        whether or not the variables should be reused. To be able to reuse the scope must be given.

    Returns
    -------
    act: (tf.Variable, bool, float) -> tf.Variable
        function to select and action given observation.
`       See the top of the file for details.
    """
    with tf.variable_scope(scope, reuse=reuse):
        observations_ph = U.ensure_tf_input(make_obs_ph("observation"))
        stochastic_ph = tf.placeholder(tf.bool, (), name="stochastic")
        update_eps_ph = tf.placeholder(tf.float32, (), name="update_eps")

        eps = tf.get_variable("eps", (), initializer=tf.constant_initializer(0))

        q_values = q_func(observations_ph.get(), num_actions, scope="q_func")
        deterministic_actions = tf.argmax(q_values, axis=1)

        batch_size = tf.shape(observations_ph.get())[0]
        random_actions = tf.random_uniform(tf.stack([batch_size]), minval=0, maxval=num_actions, dtype=tf.int64)
        chose_random = tf.random_uniform(tf.stack([batch_size]), minval=0, maxval=1, dtype=tf.float32) < eps
        stochastic_actions = tf.where(chose_random, random_actions, deterministic_actions)

        output_actions = tf.cond(stochastic_ph, lambda: stochastic_actions, lambda: deterministic_actions)
        update_eps_expr = eps.assign(tf.cond(update_eps_ph >= 0, lambda: update_eps_ph, lambda: eps))

        act = U.function(inputs=[observations_ph, stochastic_ph, update_eps_ph],
                         outputs=output_actions,
                         givens={update_eps_ph: -1.0, stochastic_ph: True},
                         updates=[update_eps_expr])
        return act


def build_train(make_obs_ph, q_func, num_actions, optimizer, grad_norm_clipping=None, gamma=1.0, double_q=True, scope="deepq", reuse=None):
    """Creates the act function:

    Parameters
    ----------
    make_obs_ph: str -> tf.placeholder or TfInput
        a function that take a name and creates a placeholder of input with that name
    q_func: (tf.Variable, int, str, bool) -> tf.Variable
        the model that takes the following inputs:
            observation_in: object
                the output of observation placeholder
            num_actions: int
                number of actions
            scope: str
            reuse: bool
                should be passed to outer variable scope
        and returns a tensor of shape (batch_size, num_actions) with values of every action.
    num_actions: int
        number of actions
    reuse: bool
        whether or not to reuse the graph variables
    optimizer: tf.train.Optimizer
        optimizer to use for the Q-learning objective.
    grad_norm_clipping: float or None
        clip graident norms to this value. If None no clipping is performed.
    gamma: float
        discount rate.
    double_q: bool
        if true will use Double Q Learning (https://arxiv.org/abs/1509.06461).
        In general it is a good idea to keep it enabled.
    scope: str or VariableScope
        optional scope for variable_scope.
    reuse: bool or None
        whether or not the variables should be reused. To be able to reuse the scope must be given.

    Returns
    -------
    act: (tf.Variable, bool, float) -> tf.Variable
        function to select and action given observation.
`       See the top of the file for details.
    train: (object, np.array, np.array, object, np.array, np.array) -> np.array
        optimize the error in Bellman's equation.
`       See the top of the file for details.
    update_target: () -> ()
        copy the parameters from optimized Q function to the target Q function.
`       See the top of the file for details.
    debug: {str: function}
        a bunch of functions to print debug data like q_values.
    """
    act_f = build_act(make_obs_ph, q_func, num_actions, scope=scope, reuse=reuse)

    with tf.variable_scope(scope, reuse=reuse):
        # set up placeholders
        obs_t_input = U.ensure_tf_input(make_obs_ph("obs_t"))
        act_t_ph = tf.placeholder(tf.int32, [None], name="action")
        rew_t_ph = tf.placeholder(tf.float32, [None], name="reward")
        obs_tp1_input = U.ensure_tf_input(make_obs_ph("obs_tp1"))
        done_mask_ph = tf.placeholder(tf.float32, [None], name="done")
        importance_weights_ph = tf.placeholder(tf.float32, [None], name="weight")

        # q network evaluation
        q_t = q_func(obs_t_input.get(), num_actions, scope="q_func", reuse=True)  # reuse parameters from act
        q_func_vars = U.scope_vars(U.absolute_scope_name("q_func"))

        # target q network evalution
        q_tp1 = q_func(obs_tp1_input.get(), num_actions, scope="target_q_func")
        target_q_func_vars = U.scope_vars(U.absolute_scope_name("target_q_func"))

        # q scores for actions which we know were selected in the given state.
        q_t_selected = tf.reduce_sum(q_t * tf.one_hot(act_t_ph, num_actions), 1)

        # compute estimate of best possible value starting from state at t + 1
        if double_q:
            q_tp1_using_online_net = q_func(obs_tp1_input.get(), num_actions, scope="q_func", reuse=True)
            q_tp1_best_using_online_net = tf.arg_max(q_tp1_using_online_net, 1)
            q_tp1_best = tf.reduce_sum(q_tp1 * tf.one_hot(q_tp1_best_using_online_net, num_actions), 1)
        else:
            q_tp1_best = tf.reduce_max(q_tp1, 1)
        q_tp1_best_masked = (1.0 - done_mask_ph) * q_tp1_best

        # compute RHS of bellman equation
        q_t_selected_target = rew_t_ph + gamma * q_tp1_best_masked

        # compute the error (potentially clipped)
        td_error = q_t_selected - tf.stop_gradient(q_t_selected_target)
        errors = U.huber_loss(td_error)
        weighted_error = tf.reduce_mean(importance_weights_ph * errors)
        # compute optimization op (potentially with gradient clipping)
        if grad_norm_clipping is not None:
            optimize_expr = U.minimize_and_clip(optimizer,
                                                weighted_error,
                                                var_list=q_func_vars,
                                                clip_val=grad_norm_clipping)
        else:
            optimize_expr = optimizer.minimize(weighted_error, var_list=q_func_vars)

        # update_target_fn will be called periodically to copy Q network to target Q network
        update_target_expr = []
        for var, var_target in zip(sorted(q_func_vars, key=lambda v: v.name),
                                   sorted(target_q_func_vars, key=lambda v: v.name)):
            update_target_expr.append(var_target.assign(var))
        update_target_expr = tf.group(*update_target_expr)

        # Create callable functions
        train = U.function(
            inputs=[
                obs_t_input,
                act_t_ph,
                rew_t_ph,
                obs_tp1_input,
                done_mask_ph,
                importance_weights_ph
            ],
            outputs=td_error,
            updates=[optimize_expr]
        )
        update_target = U.function([], [], updates=[update_target_expr])

        q_values = U.function([obs_t_input], q_t)

        return act_f, train, update_target, {'q_values': q_values}


def build_train_q_aux(make_obs_ph, q_func, num_actions, optimizer, grad_norm_clipping=None, gamma=1.0, double_q=True, scope="deepq", reuse=None):
    """Creates the act function and state functions together with train_ops for the auxiliary tasks experiment.

    Parameters
    ----------
    make_obs_ph: str -> tf.placeholder or TfInput
        a function that take a name and creates a placeholder of input with that name
    q_func: (tf.Variable, int, str, bool) -> tf.Variable
        the model that takes the following inputs:
            observation_in: object
                the output of observation placeholder
            num_actions: int
                number of actions
            scope: str
            reuse: bool
                should be passed to outer variable scope
        and returns a tensor of shape (batch_size, num_actions) with values of every action.
    num_actions: int
        number of actions
    reuse: bool
        whether or not to reuse the graph variables
    optimizer: tf.train.Optimizer
        optimizer to use for the Q-learning objective.
    grad_norm_clipping: float or None
        clip graident norms to this value. If None no clipping is performed.
    gamma: float
        discount rate.
    double_q: bool
        if true will use Double Q Learning (https://arxiv.org/abs/1509.06461).
        In general it is a good idea to keep it enabled.
    scope: str or VariableScope
        optional scope for variable_scope.
    reuse: bool or None
        whether or not the variables should be reused. To be able to reuse the scope must be given.

    Returns
    -------
    act: (tf.Variable, bool, float) -> tf.Variable
        function to select and action given observation.
`       See the top of the file for details.
    train: (object, np.array, np.array, object, np.array, np.array) -> np.array
        optimize the error in Bellman's equation.
`       See the top of the file for details.
    update_target: () -> ()
        copy the parameters from optimized Q function to the target Q function.
`       See the top of the file for details.
    debug: {str: function}
        a bunch of functions to print debug data like q_values.
    """
    act_f, state_f = build_act_state(make_obs_ph, q_func, num_actions, scope=scope, reuse=reuse)

    with tf.variable_scope(scope, reuse=reuse):
        # set up placeholders
        obs_t_input = U.ensure_tf_input(make_obs_ph("obs_t"))
        act_t_ph = tf.placeholder(tf.int32, [None], name="action")
        rew_t_ph = tf.placeholder(tf.float32, [None], name="reward")
        obs_tp1_input = U.ensure_tf_input(make_obs_ph("obs_tp1"))
        done_mask_ph = tf.placeholder(tf.float32, [None], name="done")
        importance_weights_ph = tf.placeholder(tf.float32, [None], name="weight")

        # q network evaluation
        q_t, _ = q_func(obs_t_input.get(), num_actions, scope="q_func", reuse=True)  # reuse parameters from act
        q_func_vars = U.scope_vars(U.absolute_scope_name("q_func"))

        # target q network evalution
        q_tp1, _ = q_func(obs_tp1_input.get(), num_actions, scope="target_q_func")
        target_q_func_vars = U.scope_vars(U.absolute_scope_name("target_q_func"))

        # q scores for actions which we know were selected in the given state.
        q_t_selected = tf.reduce_sum(q_t * tf.one_hot(act_t_ph, num_actions), 1)

        # compute estimate of best possible value starting from state at t + 1
        if double_q:
            q_tp1_using_online_net, _ = q_func(obs_tp1_input.get(), num_actions, scope="q_func", reuse=True)
            q_tp1_best_using_online_net = tf.arg_max(q_tp1_using_online_net, 1)
            q_tp1_best = tf.reduce_sum(q_tp1 * tf.one_hot(q_tp1_best_using_online_net, num_actions), 1)
        else:
            q_tp1_best = tf.reduce_max(q_tp1, 1)
        q_tp1_best_masked = (1.0 - done_mask_ph) * q_tp1_best

        # compute RHS of bellman equation
        q_t_selected_target = rew_t_ph + gamma * q_tp1_best_masked

        # compute the error (potentially clipped)
        #with tf.name_scope('td_error'):
        td_error = q_t_selected - tf.stop_gradient(q_t_selected_target)
        errors = U.huber_loss(td_error)
        weighted_error = tf.reduce_mean(importance_weights_ph * errors)
        #tf.summary.scalar('td_error', weighted_error)
        # compute optimization op (potentially with gradient clipping)
        if grad_norm_clipping is not None:
            optimize_expr = U.minimize_and_clip(optimizer,
                                                weighted_error,
                                                var_list=q_func_vars,
                                                clip_val=grad_norm_clipping)
        else:
            optimize_expr = optimizer.minimize(weighted_error, var_list=q_func_vars)

        # update_target_fn will be called periodically to copy Q network to target Q network
        update_target_expr = []
        for var, var_target in zip(sorted(q_func_vars, key=lambda v: v.name),
                                   sorted(target_q_func_vars, key=lambda v: v.name)):
            update_target_expr.append(var_target.assign(var))
        update_target_expr = tf.group(*update_target_expr)

        # Create callable functions
        train = U.function(
            inputs=[
                obs_t_input,
                act_t_ph,
                rew_t_ph,
                obs_tp1_input,
                done_mask_ph,
                importance_weights_ph
            ],
            outputs=[errors],
            updates=[optimize_expr]
        )
        update_target = U.function([], [], updates=[update_target_expr])

        q_values = U.function([obs_t_input], q_t)

        return act_f, state_f, train, update_target, {'q_values': q_values}


def build_train_q_aux_joint(make_obs_ph, q_cnn, q_postcnn, aux_postcnn,
                            num_actions,
                            optimizer, grad_norm_clipping=None,
                            dim_state=512,
                            l=1e-1, alpha_1=1, alpha_2=1, gamma=1.0,
                            double_q=True, dueling=True,
                            scope="deepq", reuse=None):
    """Creates the act function and state functions together with train_ops for the auxiliary tasks experiment.
    The same as the original train_q_aux except 1) the optimization expression is putting together for main and the other 2 auxiliary tasks,

    """
    act_f, state_f, spred, rpred = build_predictors_joint(make_obs_ph,
                                                          q_cnn, q_postcnn, aux_postcnn,
                                                          num_actions, scope="deepq",
                                                          reuse=reuse)

    with tf.variable_scope(scope, reuse=reuse):
        # set up placeholders
        obs_t_input = U.ensure_tf_input(make_obs_ph("obs_t"))
        act_t_ph = tf.placeholder(tf.int32, [None], name="action")
        rew_t_ph = tf.placeholder(tf.float32, [None], name="reward_t")
        obs_tp1_input = U.ensure_tf_input(make_obs_ph("obs_tp1"))
        done_mask_ph = tf.placeholder(tf.float32, [None], name="done")
        importance_weights_ph = tf.placeholder(tf.float32, [None], name="weight")

        # Graph_part1: Deep Q-Network
        # q network evaluation
        with tf.variable_scope('q_func'): # (reusing the defined scope)
            state_t = q_cnn(obs_t_input.get(), dim_state=dim_state, scope="cnn_feature", reuse=True)
            # state_tp1 = q_cnn(obs_tp1_input.get(), dim_state=dim_state, scope="cnn_feature", reuse=True) # For training protocol: learning state-prediction using current network supervision
            q_t = q_postcnn(state_t, num_actions, scope="q_func_post_cnn", reuse=True)
        q_func_vars_w_aux = U.scope_vars(U.absolute_scope_name("q_func")) #  The final list of variables for optimization, (TODO) variables for training (to check if this is correct)
        q_func_vars = [var for var in q_func_vars_w_aux if 'aux' not in var.name] # The variables in q-network for copying to target network

        # target q network evalution
        with tf.variable_scope('target_q_func'):
            state_tp1_target = q_cnn(obs_tp1_input.get(), dim_state=dim_state, scope="cnn_feature") # state-prediction learning using target network supervision
            q_tp1 = q_postcnn(state_tp1_target, num_actions, scope="q_func_post_cnn")
        target_q_func_vars = U.scope_vars(U.absolute_scope_name("target_q_func"))

        # q scores for actions which we know were selected in the given state.
        q_t_selected = tf.reduce_sum(q_t * tf.one_hot(act_t_ph, num_actions), 1)

        # compute estimate of best possible value starting from state at t + 1
        if double_q:
            with tf.variable_scope('q_func'):  # (reusing the defined scope)
                state_tp1_using_online_net = q_cnn(obs_tp1_input.get(), dim_state=dim_state, scope="cnn_feature", reuse=True)
                q_tp1_using_online_net = q_postcnn(state_tp1_using_online_net, num_actions, scope="q_func_post_cnn", reuse=True)
            q_tp1_best_using_online_net = tf.arg_max(q_tp1_using_online_net, 1)
            q_tp1_best = tf.reduce_sum(q_tp1 * tf.one_hot(q_tp1_best_using_online_net, num_actions), 1)
        else:
            q_tp1_best = tf.reduce_max(q_tp1, 1)
        q_tp1_best_masked = (1.0 - done_mask_ph) * q_tp1_best

        # compute RHS of bellman equation
        q_t_selected_target = rew_t_ph + gamma * q_tp1_best_masked

        # compute the error (potentially clipped)
        td_error = q_t_selected - tf.stop_gradient(q_t_selected_target)
        errors_q = U.huber_loss(td_error)

        # Graph_part2: Auxiliary Task
        with tf.variable_scope('q_func'):
            sh_tp1, rh_t = aux_postcnn(state_t, act_t_ph, num_actions, scope='aux_func_post_cnn', reuse=True)

        spred_error = sh_tp1 - tf.stop_gradient(state_tp1_target)
        # spred_error = sh_tp1 - tf.stop_gradient(state_tp1) # (TODO) adding the option of using current_net to learn
        rpred_error = rh_t - tf.squeeze(rew_t_ph)
        rpred_error = tf.squeeze(rpred_error)[0, :]  # (TODO) it seems solving the dim-mismatch problem, but is there better solution?

        weight_action_on_state = [var for var in tf.trainable_variables() if var.name=="deepq/q_func/aux_func_post_cnn/spred_func/fully_connected_1/weights:0"]
        l1_reg = tf.norm(weight_action_on_state, ord=1) # (TODO) adding the option of choosing l-p norm with arbitrary p (easy)
        errors_spred = tf.norm(spred_error, axis=1)  # euclidean distance btw the predicted and the true states
        errors_rpred = U.huber_loss(rpred_error)  # (TODO) check if this huber error function is appropriate

        # Training Ops: Compute the final error term
        errors = errors_q + alpha_1 * (errors_spred + l * l1_reg) + alpha_2 * (errors_rpred + l * l1_reg)
        weighted_error = tf.reduce_mean(importance_weights_ph * errors)

        # compute optimization op (potentially with gradient clipping)
        if grad_norm_clipping is not None:
            optimize_expr = U.minimize_and_clip(optimizer,
                                                weighted_error,
                                                var_list=q_func_vars_w_aux,
                                                clip_val=grad_norm_clipping)
        else:
            optimize_expr = optimizer.minimize(weighted_error, var_list=q_func_vars_w_aux)

        # update_target_fn will be called periodically to copy Q network to target Q network
        update_target_expr = []
        for var, var_target in zip(sorted(q_func_vars, key=lambda v: v.name),
                                   sorted(target_q_func_vars, key=lambda v: v.name)):
            update_target_expr.append(var_target.assign(var)) # this is where the second q_func is defined!
        update_target_expr = tf.group(*update_target_expr)

        # Create callable functions
        train = U.function(
            inputs=[
                obs_t_input,
                act_t_ph,
                rew_t_ph,
                obs_tp1_input,
                done_mask_ph,
                importance_weights_ph
            ],
            outputs=[errors, errors_q, errors_spred, errors_rpred, l1_reg],
            updates=[optimize_expr]
        )
        update_target = U.function([], [], updates=[update_target_expr])

        q_values = U.function([obs_t_input], q_t)

        return act_f, state_f, rpred, spred, train, update_target, {'q_values': q_values}


def build_train_spred(make_obs_ph, spred_func, num_actions, num_outputs, optimizer, grad_norm_clipping=None, scope="deepq", reuse=None, dueling=True):
    """Creates the state prediction function and train_op:
    """
    spred_f = build_aux(make_obs_ph,
                        spred_func,
                        num_actions=num_actions,
                        num_outputs=num_outputs,
                        scope=scope,
                        name_aux='spred', reuse=reuse, dueling=dueling)
    """Introduction TBA
        Note that scope_aux is for checking if the optimization procedure is doing the right thing!
    """

    with tf.variable_scope(scope, reuse=reuse):
        # set up placeholders
        obs_t_input = U.ensure_tf_input(make_obs_ph("obs_t"))  # (TODO) 2 s_t here may be confused with obs_t, in act so have a look at tensorboard later
        s_tp1_ph = tf.placeholder(tf.float32, [None, 512], name="s_tp1")
        act_t_ph = tf.placeholder(tf.int32, [None], name="action")
        importance_weights_ph = tf.placeholder(tf.float32, [None], name="weight")

        # state-prediction and relevant varialbe identification
        sh_tp1 = spred_func(obs_t_input.get(), act_t_ph,
                            num_actions, num_outputs=512,
                            scope="spred_func", name_aux='spred', reuse=True,
                            dueling=dueling)  # predicted s in the next step
        spred_func_vars = U.scope_vars(U.absolute_scope_name("spred_func"))

        # compute the error (potentially clipped)
        with tf.name_scope('spred_error'):
            spred_error = s_tp1_ph - sh_tp1
            l = tf.constant(10, tf.float32)
            l1_reg = tf.norm(tf.trainable_variables()[-6], ord=1) + tf.norm(tf.trainable_variables()[-8], ord=1) # (TODO) if not working correctly, check the parameter selection (checked once)
            errors = tf.norm(spred_error, axis=1)  # euclidean distance btw the predicted and the true states
            weighted_error = tf.reduce_mean(importance_weights_ph * errors) + l * l1_reg
        tf.summary.scalar('spred_error', weighted_error)

        # compute optimization op (potentially with gradient clipping)
        if grad_norm_clipping is not None:
            optimize_expr = U.minimize_and_clip(optimizer,
                                                weighted_error,
                                                var_list=spred_func_vars,
                                                clip_val=grad_norm_clipping,
                                                )
        else:
            optimize_expr = optimizer.minimize(weighted_error, var_list=spred_func_vars)


        # Create callable functions
        train = U.function(
            inputs=[
                obs_t_input,
                act_t_ph,
                s_tp1_ph, # note that this is the learned rep whereas obs_t_input is raw
                importance_weights_ph
            ],
            outputs=[errors],
            updates=[optimize_expr]
        )

        # shs = U.function([s_t_ph, act_t_ph], sh_tp1) # (TODO) 2 check if this is necessary

        return spred_f, train


def build_train_rpred(make_obs_ph, rpred_func, num_actions, optimizer, grad_norm_clipping=None, scope="deepq", reuse=None, dueling=False):
    """Creates the act function:
    """
    rpred_f = build_aux(make_obs_ph,
                        rpred_func,
                        num_actions=num_actions,
                        num_outputs=1,
                        scope=scope, name_aux='rpred', reuse=reuse, dueling=dueling)

    with tf.variable_scope(scope, reuse=reuse):
        # set up placeholders
        obs_t_input = U.ensure_tf_input(make_obs_ph("obs_t"))
        act_t_ph = tf.placeholder(tf.int32, [None], name="action")
        r_t_ph = tf.placeholder(tf.float32, [None], name="r_t")
        importance_weights_ph = tf.placeholder(tf.float32, [None], name="weight")


        # reward prediction (r^_t evaluation
        rh_t = rpred_func(obs_t_input.get(), act_t_ph,
                          num_actions, num_outputs=1,
                          scope='rpred_func', name_aux='rpred', reuse=True,
                          dueling=dueling)  # predicted r to get
        rpred_func_vars = U.scope_vars(U.absolute_scope_name("rpred_func"))

        # compute the error (potentially clipped)
        with tf.name_scope('rpred_error'):
            rpred_error = tf.squeeze(r_t_ph) - rh_t
            rpred_error = tf.squeeze(rpred_error)[0, :] # TODO it seems solving the dim-mismatch problem, but is there better solution?
            errors = U.huber_loss(rpred_error)    # (TODO) check if this huber error function is appropriate
            weighted_error = tf.reduce_mean(importance_weights_ph * errors)
        tf.summary.scalar('rpred_error', weighted_error)



        # compute optimization op (potentially with gradient clipping)
        if grad_norm_clipping is not None:
            optimize_expr = U.minimize_and_clip(optimizer,
                                                weighted_error,
                                                var_list=rpred_func_vars,
                                                clip_val=grad_norm_clipping)
        else:
            optimize_expr = optimizer.minimize(weighted_error, var_list=rpred_func_vars)

        # Create callable functions
        train = U.function(
            inputs=[
                obs_t_input,
                act_t_ph,
                r_t_ph,
                importance_weights_ph
            ],
            outputs=[errors],
            updates=[optimize_expr]
        )

        # rpred_func_wrap = U.function([s_t_ph, act_t_ph], rh_t) #(TODO) to see if we need to set it as dict as build_act

        return rpred_f, train
