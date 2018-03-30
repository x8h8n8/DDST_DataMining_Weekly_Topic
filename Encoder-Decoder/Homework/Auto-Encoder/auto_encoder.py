import utils

def new_models():
    (x_train, y_train), (x_test, y_test) = utils.load_data()
    m_l, src_l = utils.train_auto_encoder(x=x_train, epochs=100, activation=linear_activation)
    m_nl, src_nl = utils.train_auto_encoder(x=x_train, epochs=100, activation=nonlinear_activation)
    return m_l, m_nl


def load_models(src_l, src_nl):
    m_l = utils.auto_encoder(linear_activation)
    m_l.load_weights(src_l)
    m_nl = utils.auto_encoder(nonlinear_activation)
    m_nl.load_weights(src_nl)
    return m_l, m_nl

if __name__ == '__main__':
    linear_activation = None
    nonlinear_activation = 'sigmoid'

    (x_train, y_train), (x_test, y_test) = utils.load_data()
    ###### select one of the following code
    # new_models()
    model_l, model_nl = load_models('your_weights_%s' % linear_activation, 'your_weights_%s' % nonlinear_activation)
    ######
    x = x_train[5]
    x_l = utils.reconstruct_x(x, model_l)
    x_nl = utils.reconstruct_x(x, model_nl)
    utils.compare(x, x_l, x_nl)

    c_l = utils.get_code(x_train, 'your_weights_%s' % linear_activation)
    c_nl = utils.get_code(x_train, 'your_weights_%s' % nonlinear_activation)
    utils.plot(c_l, y_train)
    utils.plot(c_nl, y_train)

