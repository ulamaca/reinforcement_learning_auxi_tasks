import tensorflow as tf
import tensorflow.contrib.layers as layers


def model(img_in, num_actions, scope, reuse=False):
    """As described in https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf"""
    with tf.variable_scope(scope, reuse=reuse):
        out = img_in
        with tf.variable_scope("convnet"):
            # original architecture
            out = layers.convolution2d(out, num_outputs=32, kernel_size=8, stride=4, activation_fn=tf.nn.relu)
            out = layers.convolution2d(out, num_outputs=64, kernel_size=4, stride=2, activation_fn=tf.nn.relu)
            out = layers.convolution2d(out, num_outputs=64, kernel_size=3, stride=1, activation_fn=tf.nn.relu)
        out = layers.flatten(out)

        with tf.variable_scope("action_value"):
            out = layers.fully_connected(out, num_outputs=512, activation_fn=tf.nn.relu)
            state = out # the internal state representation
            out = layers.fully_connected(out, num_outputs=num_actions, activation_fn=None)

        return out, state


def dueling_model(img_in, num_actions, scope, reuse=False):
    """As described in https://arxiv.org/abs/1511.06581"""
    with tf.variable_scope(scope, reuse=reuse):
        out = img_in
        with tf.variable_scope("convnet"):
            # original architecture
            out = layers.convolution2d(out, num_outputs=32, kernel_size=8, stride=4, activation_fn=tf.nn.relu)
            out = layers.convolution2d(out, num_outputs=64, kernel_size=4, stride=2, activation_fn=tf.nn.relu)
            out = layers.convolution2d(out, num_outputs=64, kernel_size=3, stride=1, activation_fn=tf.nn.relu)
        out = layers.flatten(out)

        with tf.variable_scope("state_value"):
            state_hidden = layers.fully_connected(out, num_outputs=512, activation_fn=tf.nn.relu)
            state = state_hidden # I choose the state for V(s) as my state for training
            state_score = layers.fully_connected(state_hidden, num_outputs=1, activation_fn=None)
        with tf.variable_scope("action_value"):
            actions_hidden = layers.fully_connected(out, num_outputs=512, activation_fn=tf.nn.relu)
            action_scores = layers.fully_connected(actions_hidden, num_outputs=num_actions, activation_fn=None)
            action_scores_mean = tf.reduce_mean(action_scores, 1)
            action_scores = action_scores - tf.expand_dims(action_scores_mean, 1)

        return state_score + action_scores, state


def auxiliary_model(img_in, action_in, num_actions, num_outputs, scope, name_aux, reuse=False, dueling=True):
        # Auxiliary task network: Using single-layer perceptron after the state for both auxiliary tasks
        # This structure is not allowed to change in the current code.
    with tf.variable_scope(scope, reuse=reuse):
        out = img_in
        with tf.variable_scope("convnet"):
            # original architecture
            out = layers.convolution2d(out, num_outputs=32, kernel_size=8, stride=4, activation_fn=tf.nn.relu)
            out = layers.convolution2d(out, num_outputs=64, kernel_size=4, stride=2, activation_fn=tf.nn.relu)
            out = layers.convolution2d(out, num_outputs=64, kernel_size=3, stride=1, activation_fn=tf.nn.relu)
        out = layers.flatten(out)

        if dueling: # the scope for hidden states is different depending on whether using dueling structure
            scope_state = "action_value"
        else:
            scope_state = "state_value"

        with tf.variable_scope(scope_state):
            out = layers.fully_connected(out, num_outputs=512, activation_fn=tf.nn.relu)

        action = tf.one_hot(action_in, depth=num_actions) # convert action-label from int to one-hot coding
        with tf.variable_scope(name_aux):
            out_a = layers.fully_connected(action, num_outputs=256, activation_fn=tf.nn.relu)
            out_a = layers.fully_connected(out_a, num_outputs=num_outputs, activation_fn=None)
            out_s = layers.fully_connected(out, num_outputs=256, activation_fn=tf.nn.relu)
            out_s = layers.fully_connected(out_s, num_outputs=num_outputs, activation_fn=None)
            out_aux = out_a + out_s

        return out_aux