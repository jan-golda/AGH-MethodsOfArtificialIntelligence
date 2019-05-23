import random

import keras.applications.mobilenet_v2 as k_mobilenet_v2
import keras.backend as k
import keras.datasets.mnist as k_mnist
import keras.layers as k_layers
import keras.losses as k_losses
import keras.models as k_models
import keras.optimizers as k_optimizers
import keras.preprocessing.image as k_image
import keras.utils as k_utils
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

data = k_mnist.load_data()
(train_images, train_labels), (test_images, test_labels) = data
batch_size = 100


def _init_session():
    """ Utility function that initializes and returns TensorFlow session """
    # configuration
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    # start session
    session = tf.Session(config=config)
    init = tf.global_variables_initializer()
    session.run(init)

    return session


def _optimise(session, x, correct_y, loss):
    """ Utility function responsible for optimalization (training) using gradient descent """
    # create optimizer
    optimizer = tf.train.GradientDescentOptimizer(0.5)
    train_step = optimizer.minimize(loss)

    # split data into batches to speed up learning process
    for i in range(train_images.shape[0] // batch_size):
        batch_train_images = train_images[i * batch_size:(i + 1) * batch_size, :, :]
        batch_train_labels = train_labels[i * batch_size:(i + 1) * batch_size]

        # start optimizer in session
        session.run(train_step, feed_dict={x: batch_train_images, correct_y: batch_train_labels})


def _test_accuracy(session, x, correct_y, y, y_):
    """ Utility function that returns how well the network was trained  """
    # calculate correct results (when the highest result is for the right answer)
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

    # calculate accuracy as percent of correct results
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # start calculations
    return session.run(accuracy, feed_dict={x: test_images, correct_y: test_labels})


def intro():
    """ Display random image alongside its label """
    i = random.randint(0, len(train_images))
    plt.imshow(train_images[i])
    plt.title(train_labels[i], fontdict={'fontweight': 'bold'})
    plt.show()


def exercise_one():
    # tworzymy nasza pierwsza, minimalna siec

    # tf.placeholder to miejsce na tensor, ktory zostanie zawsze podany przed startem obliczen i nigdy sie nie zmieni
    # z jego uzyciem najpierw przygotowujemy miejsca na wejsciowy obraz...
    # w tym przypadku nie zastepujemy None - oznaczaja one dowolny rozmiar
    x = tf.placeholder(tf.float32, [None, 28, 28])  # miejsce na obraz cyfry (28px na 28px)
    x_ = tf.reshape(x, [-1, 784])  # reshape do liniowego ksztaltu (28x28=784), -1 to automatyczne dopasowanie reszty
    x_ /= 255.0  # podzielenie przez 255.0 normalizuje wartosci do zakresu od 0.0 do 1.0
    # ...oraz etykiete oznaczajaca cyfre, jaka przedstawia
    correct_y = tf.placeholder(tf.uint8, [None])  # miejsce na klase cyfry (mozliwe 10 roznych klas)
    y_ = tf.one_hot(correct_y, 10)  # zamienia cyfrę na wektor o 10 pozycjach, tylko jedna z nich jest inna niz 0 (hot)

    # potem przygotowujemy zmienne - tensory ktorych wartosci beda ewoluowac w czasie
    # inicjujemy je zerami o wlasciwym ksztalcie - tu wlasnie znajda sie poszukiwane przez nas optymalne wagi
    w = tf.Variable(tf.zeros([784, 10]))  # wagi funkcji liniowej (10 neuronow, kazdy ma 784 wejscia)
    b = tf.Variable(tf.zeros([10]))  # bias funkcji liniowej (dla kazdego z 10 neuronow)

    # wykorzystaj przygotowane wyzej zmienne oraz tf.matmul i tf.nn.softmax
    y = tf.nn.softmax(tf.matmul(x_, w) + b)

    # wykorzystaj przygotowane wyzej zmienne oraz tf.log; zastanow sie jak zsumowac tylko wlasciwe elementy wyjscia!
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(tf.log(y) * y_, reduction_indices=[1]))

    # uruchamiamy graf obliczen tensor flow
    session = _init_session()  # startujemy sesje
    _optimise(session, x, correct_y, cross_entropy)  # optymalizujemy entropie krzyzowa
    print(f"Accuracy: {_test_accuracy(session, x, correct_y, y, y_)}")  # i weryfikujemy skutki na danych testowych


def exercise_two():
    # dodajemy do naszej sieci dodatkowa warstwe, w celu zwiekszenia jej pojemnosci i sily wyrazu

    # ta sekcja jest powtorka z poprzedniego cwiczenia - rozstawia warstwe wejsciowa
    x = tf.placeholder(tf.float32, [None, 28, 28])
    x_ = tf.reshape(x, [-1, 784]) / 255.0
    correct_y = tf.placeholder(tf.uint8, [None])
    y_ = tf.one_hot(correct_y, 10)

    # kolejne dwie sekcje nieco sie juz roznia - dodalismy dodatkowa warstwe wewnetrzna
    w1 = tf.Variable(tf.zeros([784, 100]))
    b1 = tf.Variable(tf.zeros([100]))
    h1 = tf.nn.relu(tf.matmul(x_, w1) + b1)

    w2 = tf.Variable(tf.zeros([100, 10]))
    b2 = tf.Variable(tf.zeros([10]))
    y = tf.nn.softmax(tf.matmul(h1, w2) + b2)
    # dla pierwszej warstwy skorzystaj z tf.nn.relu, dla drugiej (wyjsciowej) nadal z tf.nn.softmax

    cross_entropy = tf.reduce_mean(-tf.reduce_sum(tf.log(y) * y_, reduction_indices=[1]))

    # ta sekcja rowniez jest powtorka z poprzedniego cwiczenia
    session = _init_session()
    _optimise(session, x, correct_y, cross_entropy)
    print(f"Accuracy: {_test_accuracy(session, x, correct_y, y, y_)}")


def exercise_three():
    # zmien poczatkowe wartosci wag na losowe z odchyleniem standardowym od -0.1 do 0.1, a biasy na po prostu 0.1
    # skorzystaj z funkcji tf.truncated_normal (dla wag) i tf.constant (dla biasow)

    x = tf.placeholder(tf.float32, [None, 28, 28])
    x_ = tf.reshape(x, [-1, 784]) / 255.0
    correct_y = tf.placeholder(tf.uint8, [None])
    y_ = tf.one_hot(correct_y, 10)

    w1 = tf.Variable(tf.truncated_normal([784, 100], stddev=0.1))
    b1 = tf.Variable(tf.constant(0.1, tf.float32, [100]))
    h1 = tf.nn.relu(tf.matmul(x_, w1) + b1)

    w2 = tf.Variable(tf.truncated_normal([100, 10], stddev=0.1))
    b2 = tf.Variable(tf.constant(0.1, tf.float32, [10]))
    y = tf.nn.softmax(tf.matmul(h1, w2) + b2)

    cross_entropy = tf.reduce_mean(-tf.reduce_sum(tf.log(y) * y_, reduction_indices=[1]))

    session = _init_session()
    _optimise(session, x, correct_y, cross_entropy)
    print(f"Accuracy: {_test_accuracy(session, x, correct_y, y, y_)}")


def exercise_four():
    # tym razem tworzymy siec identyczna jak poprzednia, ale wykorzystujac wysokopoziomowa biblioteke Keras
    model = k_models.Sequential()  # nasza siec bedzie skladac sie z sekwencji warstw (wymienionych ponizej)

    # klasa k_layers.X to w rzeczywistosci keras.layers.X (patrz importy u gory pliku) - analogicznie dla innych modulow
    model.add(k_layers.Reshape((784, ), input_shape=(28, 28)))  # zmienia ksztalt wejscia z (28, 28) na (784, )
    model.add(k_layers.Lambda(lambda x: x / 255.0))  # normalizuje wejscie z 0.0 do 255.0 na 0.0 do 1.0
    model.add(k_layers.Dense(units=100, input_dim=784, activation='relu'))  # pierwsza prawdziwa warstwa neuronow
    model.add(k_layers.Dense(units=10, input_dim=100, activation='softmax'))  # wyjsciowa warstwa neuronow

    # teraz kompilujemy nasz zdefiniowany wczesniej model
    model.compile(
        loss=k_losses.categorical_crossentropy,  # tu podajemy czym jest funkcja loss
        optimizer=k_optimizers.SGD(0.5),  # a tu podajemy jak ja optymalizowac
        metrics=['accuracy']  # tu informujemy, by w trakcie pracy zbierac informacje o uzyskanej skutecznosci
    )
    # trenujemy nasz skompilowany model (k_utils.to_categorical jest odpowiednikiem tf.one_hot)
    model.fit(train_images, k_utils.to_categorical(train_labels), epochs=1, batch_size=batch_size)
    # oraz ewaluujemy jego skutecznosc
    loss_and_metrics = model.evaluate(test_images, k_utils.to_categorical(test_labels))
    print(f"Final accuracy result: {loss_and_metrics[1] * 100.0} %")


def exercise_five():
    # na koniec skorzystamy z gotowej, wytrenowanej juz sieci glebokiej

    # pobranie gotowego modelu zlozonej sieci konwolucyjnej z odpowiednimi wagami
    # include_top=True oznacza ze pobieramy wszystkie warstwy - niektore zastosowania korzystaja tylko z dolnych
    model = k_mobilenet_v2.MobileNetV2(weights='imagenet', include_top=True)
    # podejrzenie tego z ilu i jakich warstw sie sklada
    layers = dict([(layer.name, layer.output) for layer in model.layers])
    print("Network containts following layers:")
    for i, (name, layer) in enumerate(layers.items()):
        print("Layer {0} : {1}".format(i, (name, layer)))
    # oraz ile parametrow musialo zostac wytrenowanych
    print("Together: {0} parameters\n".format(model.count_params()))
    # powyzsza siec jest i tak wyjatkowo mala - sluzy do zastosowan mobilnych, wiec jest silnie miniaturyzowana

    # otworzmy przykladowe zdjecie i dostosujemy jego rozmiar i zakres wartosci do wejscia sieci
    image_path = 'meat-loaf.jpeg'
    image = k_image.load_img(image_path, target_size=(224, 224))
    x = k_image.img_to_array(image)  # kolejne linie dodatkowo dostosowuja obraz pod dana siec
    x = np.expand_dims(x, axis=0)
    x = k_mobilenet_v2.preprocess_input(x)

    # sprawdzmy jaki wynik przewidzi siec
    predictions = model.predict(x)
    # i przetlumaczmy uzywajac etykiet zrozumialych dla czlowieka (5 najbardziej prawdopodobnych klas zdaniem sieci)
    print('Predicted class:', k_mobilenet_v2.decode_predictions(predictions, top=5)[0])

    # https://github.com/raghakot/keras-vis/blob/master/resources/imagenet_class_index.json

    # finalnie podgladamy aktywacje jakie wysylaja neurony sieci w trakcie dzialania
    # w wypisanych wczesniej informacjach mozna latwo spradzic ile kanalow ma warstwa o danym numerze (i ktora to)
    layer_to_preview = 10  # numer warstwy, ktorej aktywacje podgladamy
    channel_to_preview = 16   # numer kanalu w tejze warstwie
    get_activations = k.function([model.layers[0].input], [model.layers[layer_to_preview].output])
    activations = get_activations([x])
    plt.imshow(activations[0][0, :, :, channel_to_preview], cmap="viridis")
    plt.show()


def main():
    # intro()
    # exercise_one()
    # exercise_two()
    # exercise_three()
    # exercise_four()
    exercise_five()


if __name__ == '__main__':
    main()
