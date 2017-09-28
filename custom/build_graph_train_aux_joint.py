def build_train_q_aux_joint(make_obs_ph, q_func, spred_func, rpred_func, num_outputs, num_actions,
                            optimizer, grad_norm_clipping=None,
                            l =1e-1, alpha_1=1, alpha_2=1,
                            gamma=1.0, double_q=True, dueling=True, scope="deepq", reuse=None):
    """Creates the act function and state functions together with train_ops for the auxiliary tasks experiment.
    The same as the original train_q_aux except 1) the optimization expression is putting together for main and the other 2 auxiliary tasks,

    """
    act_f, state_f = build_act_state(make_obs_ph, q_func, num_actions, scope=scope, reuse=reuse)

    with tf.variable_scope(scope, reuse=reuse):
        # set up placeholders
        obs_t_input = U.ensure_tf_input(make_obs_ph("obs_t"))
        s_tp1_ph = tf.placeholder(tf.float32, [None, 512], name="s_tp1")
        act_t_ph = tf.placeholder(tf.int32, [None], name="action")
        rew_t_ph = tf.placeholder(tf.float32, [None], name="reward")
        obs_tp1_input = U.ensure_tf_input(make_obs_ph("obs_tp1"))
        r_t_ph = tf.placeholder(tf.float32, [None], name="r_t")
        done_mask_ph = tf.placeholder(tf.float32, [None], name="done")
        importance_weights_ph = tf.placeholder(tf.float32, [None], name="weight")

        # Graph_part1: Deep Q-Network
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
        td_error = q_t_selected - tf.stop_gradient(q_t_selected_target)
        errors_Q = U.huber_loss(td_error)

        # Graph_part2: Internal State Prediction Function
        sh_tp1 = spred_func(obs_t_input.get(), act_t_ph,
                            num_actions, num_outputs=512,
                            scope="spred_func", name_aux='spred', reuse=True,
                            dueling=dueling)  # predicted s in the next step
        spred_func_vars = U.scope_vars(U.absolute_scope_name("spred_func"))

        spred_error = s_tp1_ph - sh_tp1
        l1_reg = tf.norm(tf.trainable_variables()[-6], ord=1) + tf.norm(tf.trainable_variables()[-8], ord=1) # (TODO) if not working correctly, check the parameter selection (checked once)
        errors_spred = tf.norm(spred_error, axis=1)  # euclidean distance btw the predicted and the true states

        # Graph_part3: Reward Prediction Function
        rpred_f = build_aux(make_obs_ph,
                            rpred_func,
                            num_actions=num_actions,
                            num_outputs=1,
                            scope=scope, name_aux='rpred', reuse=reuse, dueling=dueling)

        # reward prediction (r^_t evaluation)
        rh_t = rpred_func(obs_t_input.get(), act_t_ph,
                          num_actions, num_outputs=1,
                          scope='rpred_func', name_aux='rpred', reuse=True,
                          dueling=dueling)  # predicted r to get
        rpred_func_vars = U.scope_vars(U.absolute_scope_name("rpred_func"))

        # compute the error (potentially clipped)
        rpred_error = tf.squeeze(r_t_ph) - rh_t
        rpred_error = tf.squeeze(rpred_error)[0, :]  # TODO it seems solving the dim-mismatch problem, but is there better solution?
        errors_rpred = U.huber_loss(rpred_error)  # (TODO) check if this huber error function is appropriate

        # 5 Compute the final error term
        errors = errors_Q + alpha_1 * (errors_spred + l * l1_reg) + alpha_2 * errors_rpred
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
            outputs=[errors],
            updates=[optimize_expr]
        )
        update_target = U.function([], [], updates=[update_target_expr])

        q_values = U.function([obs_t_input], q_t)

        return act_f, state_f, rpred_func, spred_func, train, update_target, {'q_values': q_values}