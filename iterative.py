from __future__ import division
import numpy as np
from abc import abstractmethod
import logging
import warnings

from foolbox2 import models
from foolbox2.utils import crossentropy
from foolbox2.attacks.base import Attack, call_decorator
# from foolbox.models.base import Attack, call_decorator
from foolbox2 import distances
from foolbox2.utils import crossentropy


def find_salience(model_path, im):
    pass

class IterativeProjectedGradientBaseAttack(Attack):
    """Base class for iterative (projected) gradient attacks.

    Concrete subclasses should implement __call__, _gradient
    and _clip_perturbation.

    TODO: add support for other loss-functions, e.g. the CW loss function,
    see https://github.com/MadryLab/mnist_challenge/blob/master/pgd_attack.py
    """
    @abstractmethod
    def _gradient(self, a, x, class_, strict=True):
        raise NotImplementedError

    @abstractmethod
    def _clip_perturbation(self, a, noise, epsilon):
        raise NotImplementedError

    @abstractmethod
    def _check_distance(self, a):
        raise NotImplementedError

    def _get_mode_and_class(self, a):
        # determine if the attack is targeted or not
        target_class = a.target_class()
        targeted = target_class is not None

        if targeted:
            class_ = target_class
        else:
            class_ = a.original_class
        return targeted, class_

    def _run(self, a, binary_search,
             epsilon, stepsize, iterations,
             random_start, return_early):
        if not a.has_gradient():
            warnings.warn('applied gradient-based attack to model that'
                          ' does not provide gradients')
            return

        self._check_distance(a)

        targeted, class_ = self._get_mode_and_class(a)

        if binary_search:
            if isinstance(binary_search, bool):
                k = 20
            else:
                k = int(binary_search)
            return self._run_binary_search(
                a, epsilon, stepsize, iterations,
                random_start, targeted, class_, return_early, k=k)
        else:
            return self._run_one(
                a, epsilon, stepsize, iterations,
                random_start, targeted, class_, return_early)

    def _run_binary_search(self, a, epsilon, stepsize, iterations,
                           random_start, targeted, class_, return_early, k):

        factor = stepsize / epsilon

        def try_epsilon(epsilon):
            stepsize = factor * epsilon
            return self._run_one(
                a, epsilon, stepsize, iterations,
                random_start, targeted, class_, return_early)

        for i in range(k):
            if try_epsilon(epsilon):
                logging.info('successful for eps = {}'.format(epsilon))
                break
            logging.info('not successful for eps = {}'.format(epsilon))
            epsilon = epsilon * 1.5
        else:
            logging.warning('exponential search failed')
            return

        bad = 0
        good = epsilon

        for i in range(k):
            epsilon = (good + bad) / 2
            if try_epsilon(epsilon):
                good = epsilon
                logging.info('successful for eps = {}'.format(epsilon))
            else:
                bad = epsilon
                logging.info('not successful for eps = {}'.format(epsilon))

    def _run_one(self, a, epsilon, stepsize, iterations,
                 random_start, targeted, class_, return_early):
        min_, max_ = a.bounds()
        s = max_ - min_

        original = a.original_image.copy()

        if random_start:
            # using uniform noise even if the perturbation clipping uses
            # a different norm because cleverhans does it the same way
            noise = np.random.uniform(
                -epsilon * s, epsilon * s, original.shape).astype(
                    original.dtype)
            x = original + self._clip_perturbation(a, noise, epsilon)
            strict = False  # because we don't enforce the bounds here
        else:
            x = original
            strict = True

        success = False
        for _ in range(iterations):
            gradient = self._gradient(a, x, class_, strict=strict)
            # non-strict only for the first call and
            # only if random_start is True
            strict = True
            if targeted:
                gradient = -gradient

            # untargeted: gradient ascent on cross-entropy to original class
            # targeted: gradient descent on cross-entropy to target class
            x = x + stepsize * gradient

            x = original + self._clip_perturbation(a, x - original, epsilon)

            x = np.clip(x, min_, max_)

            logits, is_adversarial = a.predictions(x)
            if logging.getLogger().isEnabledFor(logging.DEBUG):
                if targeted:
                    ce = crossentropy(a.original_class, logits)
                    logging.debug('crossentropy to {} is {}'.format(
                        a.original_class, ce))
                ce = crossentropy(class_, logits)
                logging.debug('crossentropy to {} is {}'.format(class_, ce))
            if is_adversarial:
                if return_early:
                    return True
                else:
                    success = True
        return success


class LinfinityGradientMixin(object):
    def _gradient(self, a, x, class_, strict=True):
        gradient = a.gradient(x, class_, strict=strict)
        gradient = np.sign(gradient)
        min_, max_ = a.bounds()
        gradient = (max_ - min_) * gradient
        return gradient


class L1GradientMixin(object):
    def _gradient(self, a, x, class_, strict=True):
        gradient = a.gradient(x, class_, strict=strict)
        # using mean to make range of epsilons comparable to Linf
        gradient = gradient / np.mean(np.abs(gradient))
        min_, max_ = a.bounds()
        gradient = (max_ - min_) * gradient
        return gradient


class L2GradientMixin(object):
    def _gradient(self, a, x, class_, strict=True):
        gradient = a.gradient(x, class_, strict=strict)
        # using mean to make range of epsilons comparable to Linf
        gradient = gradient / np.sqrt(np.mean(np.square(gradient)))
        min_, max_ = a.bounds()
        gradient = (max_ - min_) * gradient
        return gradient


class LinfinityClippingMixin(object):
    def _clip_perturbation(self, a, perturbation, epsilon):
        min_, max_ = a.bounds()
        s = max_ - min_
        clipped = np.clip(perturbation, -epsilon * s, epsilon * s)
        return clipped


class L1ClippingMixin(object):
    def _clip_perturbation(self, a, perturbation, epsilon):
        # using mean to make range of epsilons comparable to Linf
        norm = np.mean(np.abs(perturbation))
        norm = max(1e-12, norm)  # avoid divsion by zero
        min_, max_ = a.bounds()
        s = max_ - min_
        # clipping, i.e. only decreasing norm
        factor = min(1, epsilon * s / norm)
        return perturbation * factor


class L2ClippingMixin(object):
    def _clip_perturbation(self, a, perturbation, epsilon):
        # using mean to make range of epsilons comparable to Linf
        norm = np.sqrt(np.mean(np.square(perturbation)))
        norm = max(1e-12, norm)  # avoid divsion by zero
        min_, max_ = a.bounds()
        s = max_ - min_
        # clipping, i.e. only decreasing norm
        factor = min(1, epsilon * s / norm)
        return perturbation * factor


class LinfinityDistanceCheckMixin(object):
    def _check_distance(self, a):
        if not isinstance(a.distance, distances.Linfinity):
            logging.warning('Running an attack that tries to minimize the'
                            ' Linfinity norm of the perturbation without'
                            ' specifying foolbox.distances.Linfinity as'
                            ' the distance metric might lead to suboptimal'
                            ' results.')


class L1DistanceCheckMixin(object):
    def _check_distance(self, a):
        if not isinstance(a.distance, distances.MAE):
            logging.warning('Running an attack that tries to minimize the'
                            ' L1 norm of the perturbation without'
                            ' specifying foolbox.distances.MAE as'
                            ' the distance metric might lead to suboptimal'
                            ' results.')


class L2DistanceCheckMixin(object):
    def _check_distance(self, a):
        if not isinstance(a.distance, distances.MSE):
            logging.warning('Running an attack that tries to minimize the'
                            ' L2 norm of the perturbation without'
                            ' specifying foolbox.distances.MSE as'
                            ' the distance metric might lead to suboptimal'
                            ' results.')


class IterativeProjectedGradientBaseAttack(Attack):
    """Base class for iterative (projected) gradient attacks.
    Concrete subclasses should implement __call__, _gradient
    and _clip_perturbation.
    TODO: add support for other loss-functions, e.g. the CW loss function,
    see https://github.com/MadryLab/mnist_challenge/blob/master/pgd_attack.py
    """
    @abstractmethod
    def _gradient(self, a, x, class_, strict=True):
        raise NotImplementedError

    @abstractmethod
    def _clip_perturbation(self, a, noise, epsilon):
        raise NotImplementedError

    @abstractmethod
    def _check_distance(self, a):
        raise NotImplementedError

    def _get_mode_and_class(self, a):
        # determine if the attack is targeted or not
        target_class = a.target_class()
        targeted = target_class is not None

        if targeted:
            class_ = target_class
        else:
            class_ = a.original_class
        return targeted, class_

    def _run(self, a, positive_salience, binary_search,
             epsilon, stepsize, iterations,
             random_start, return_early):
        if not a.has_gradient():
            warnings.warn('applied gradient-based attack to model that'
                          ' does not provide gradients')
            return

        self._check_distance(a)

        targeted, class_ = self._get_mode_and_class(a)

        if binary_search:
            if isinstance(binary_search, bool):
                k = 20
            else:
                k = int(binary_search)
            return self._run_binary_search(
                a, positive_salience, epsilon, stepsize, iterations,
                random_start, targeted, class_, return_early, k=k)
        else:
            return self._run_one(
                a, positive_salience, epsilon, stepsize, iterations,
                random_start, targeted, class_, return_early)

    def _run_binary_search(self, a, positive_salience, epsilon, stepsize, iterations,
                           random_start, targeted, class_, return_early, k):

        factor = stepsize / epsilon

        def try_epsilon(epsilon):
            stepsize = factor * epsilon
            return self._run_one(
                a, positive_salience, epsilon, stepsize, iterations,
                random_start, targeted, class_, return_early)

        for i in range(k):
            if try_epsilon(epsilon):
                logging.info('successful for eps = {}'.format(epsilon))
                break
            logging.info('not successful for eps = {}'.format(epsilon))
            epsilon = epsilon * 1.5
        else:
            logging.warning('exponential search failed')
            return

        bad = 0
        good = epsilon

        for i in range(k):
            epsilon = (good + bad) / 2
            if try_epsilon(epsilon):
                good = epsilon
                logging.info('successful for eps = {}'.format(epsilon))
            else:
                bad = epsilon
                logging.info('not successful for eps = {}'.format(epsilon))

    def _run_one(self, a, positive_salience, epsilon, stepsize, iterations,
                 random_start, targeted, class_, return_early):
        min_, max_ = a.bounds()
        s = max_ - min_
        original = a.original_image.copy()
        if random_start:
            # using uniform noise even if the perturbation clipping uses
            # a different norm because cleverhans does it the same way
            noise = np.random.uniform(
                -epsilon * s, epsilon * s, original.shape).astype(
                    original.dtype)
            x = original + self._clip_perturbation(a, noise, epsilon)
            strict = False  # because we don't enforce the bounds here
        else:
            x = original
            strict = True

        success = False
        for _ in range(iterations):
            gradient = self._gradient(a, x, class_, strict=strict)
            # non-strict only for the first call and
            # only if random_start is True
            strict = True
            if targeted:
                gradient = -gradient

            # untargeted: gradient ascent on cross-entropy to original class
            # targeted: gradient descent on cross-entropy to target class
            x = x + stepsize * gradient

            x = original + self._clip_perturbation(a, x - original, epsilon)

            x = np.clip(x, min_, max_)

            if positive_salience is not None:
                and_with_pos = np.logical_and(positive_salience, x)
                noise_mask = and_with_pos.astype(int)
                original_mask = abs(1 - noise_mask)
                x = x * noise_mask + original * original_mask

            logits, is_adversarial = a.predictions(x)
            if logging.getLogger().isEnabledFor(logging.DEBUG):
                if targeted:
                    ce = crossentropy(a.original_class, logits)
                    logging.debug('crossentropy to {} is {}'.format(
                        a.original_class, ce))
                ce = crossentropy(class_, logits)
                logging.debug('crossentropy to {} is {}'.format(class_, ce))
            if is_adversarial:
                if return_early:
                    return True
                else:
                    success = True
        return success


class LinfinityGradientMixin(object):
    def _gradient(self, a, x, class_, strict=True):
        gradient = a.gradient(x, class_, strict=strict)
        gradient = np.sign(gradient)
        min_, max_ = a.bounds()
        gradient = (max_ - min_) * gradient
        return gradient


class L1GradientMixin(object):
    def _gradient(self, a, x, class_, strict=True):
        gradient = a.gradient(x, class_, strict=strict)
        # using mean to make range of epsilons comparable to Linf
        gradient = gradient / np.mean(np.abs(gradient))
        min_, max_ = a.bounds()
        gradient = (max_ - min_) * gradient
        return gradient


class L2GradientMixin(object):
    def _gradient(self, a, x, class_, strict=True):
        gradient = a.gradient(x, class_, strict=strict)
        # using mean to make range of epsilons comparable to Linf
        gradient = gradient / np.sqrt(np.mean(np.square(gradient)))
        min_, max_ = a.bounds()
        gradient = (max_ - min_) * gradient
        return gradient


class LinfinityClippingMixin(object):
    def _clip_perturbation(self, a, perturbation, epsilon):
        min_, max_ = a.bounds()
        s = max_ - min_
        clipped = np.clip(perturbation, -epsilon * s, epsilon * s)
        return clipped


class L1ClippingMixin(object):
    def _clip_perturbation(self, a, perturbation, epsilon):
        # using mean to make range of epsilons comparable to Linf
        norm = np.mean(np.abs(perturbation))
        norm = max(1e-12, norm)  # avoid divsion by zero
        min_, max_ = a.bounds()
        s = max_ - min_
        # clipping, i.e. only decreasing norm
        factor = min(1, epsilon * s / norm)
        return perturbation * factor


class L2ClippingMixin(object):
    def _clip_perturbation(self, a, perturbation, epsilon):
        # using mean to make range of epsilons comparable to Linf
        norm = np.sqrt(np.mean(np.square(perturbation)))
        norm = max(1e-12, norm)  # avoid divsion by zero
        min_, max_ = a.bounds()
        s = max_ - min_
        # clipping, i.e. only decreasing norm
        factor = min(1, epsilon * s / norm)
        return perturbation * factor


class LinfinityDistanceCheckMixin(object):
    def _check_distance(self, a):
        if not isinstance(a.distance, distances.Linfinity):
            logging.warning('Running an attack that tries to minimize the'
                            ' Linfinity norm of the perturbation without'
                            ' specifying foolbox.distances.Linfinity as'
                            ' the distance metric might lead to suboptimal'
                            ' results.')


class L1DistanceCheckMixin(object):
    def _check_distance(self, a):
        if not isinstance(a.distance, distances.MAE):
            logging.warning('Running an attack that tries to minimize the'
                            ' L1 norm of the perturbation without'
                            ' specifying foolbox.distances.MAE as'
                            ' the distance metric might lead to suboptimal'
                            ' results.')


class L2DistanceCheckMixin(object):
    def _check_distance(self, a):
        if not isinstance(a.distance, distances.MSE):
            logging.warning('Running an attack that tries to minimize the'
                            ' L2 norm of the perturbation without'
                            ' specifying foolbox.distances.MSE as'
                            ' the distance metric might lead to suboptimal'
                            ' results.')

class SAIterativeAttack(
        L2ClippingMixin,
        L2DistanceCheckMixin,
        IterativeProjectedGradientBaseAttack):
    def _gradient(self, a, x, class_, strict=True):
        # get current gradient
        gradient = a.gradient(x, class_, strict=strict)
        gradient = gradient / max(1e-12, np.mean(np.abs(gradient)))

        # combine with history of gradient as new history
        self._momentum_history = \
            self._decay_factor * self._momentum_history + gradient

        # use history
        gradient = self._momentum_history
        gradient = np.sign(gradient)
        min_, max_ = a.bounds()
        gradient = (max_ - min_) * gradient
        return gradient

    def _run_one(self, *args, **kwargs):
        # reset momentum history every time we restart
        # gradient descent
        self._momentum_history = 0
        return super(SAIterativeAttack, self)._run_one(*args, **kwargs)

    @call_decorator
    def __call__(self, input_or_adv, model_path = None, label=None, unpack=True,
                 binary_search=True,
                 epsilon=0.3,
                 stepsize=0.06,
                 iterations=10,
                 decay_factor=1.0,
                 random_start=False,
                 return_early=True):

        """Momentum-based iterative gradient attack known as
        Momentum Iterative Method.
        Parameters
        ----------
        input_or_adv : `numpy.ndarray` or :class:`Adversarial`
            The original, unperturbed input as a `numpy.ndarray` or
            an :class:`Adversarial` instance.
        label : int
            The reference label of the original input. Must be passed
            if `a` is a `numpy.ndarray`, must not be passed if `a` is
            an :class:`Adversarial` instance.
        unpack : bool
            If true, returns the adversarial input, otherwise returns
            the Adversarial object.
        binary_search : bool
            Whether to perform a binary search over epsilon and stepsize,
            keeping their ratio constant and using their values to start
            the search. If False, hyperparameters are not optimized.
            Can also be an integer, specifying the number of binary
            search steps (default 20).
        epsilon : float
            Limit on the perturbation size; if binary_search is True,
            this value is only for initialization and automatically
            adapted.
        stepsize : float
            Step size for gradient descent; if binary_search is True,
            this value is only for initialization and automatically
            adapted.
        iterations : int
            Number of iterations for each gradient descent run.
        decay_factor : float
            Decay factor used by the momentum term.
        random_start : bool
            Start the attack from a random point rather than from the
            original input.
        return_early : bool
            Whether an individual gradient descent run should stop as
            soon as an adversarial is found.
        """
        a = input_or_adv
        del input_or_adv
        del label
        del unpack

        assert epsilon > 0

        self._decay_factor = decay_factor
        # self._initial_temperature = _initial_temperature
        original = a.original_image.copy()
        positive_salience = None
        if model_path:
            positive_salience = find_salience(model_path, original)

        self._run(a, positive_salience, binary_search,
                  epsilon, stepsize, iterations,
                  random_start, return_early)

class RMSIterativeAttack(
        L2ClippingMixin,
        L2DistanceCheckMixin,
        IterativeProjectedGradientBaseAttack):
    def _gradient(self, a, x, class_, strict=True):
        # get current gradient
        gradient = a.gradient(x, class_, strict=strict)
        noise = gradient / max(1e-12, np.mean(np.abs(gradient)))
        # gradient = np.multiply(gradient, gradient)
        # combine with history of gradient as new history
        # the new noise becomes the momentum history
        # use history
        noise = noise * noise;
        noise = self._gamma * self._gradient_history + (1 - self._gamma) * noise
        self._gradient_history  = noise
        #det = np.clip(np.round(noise), 0, 1) - 0.5
        noise = self._alpha/np.sqrt(noise) * gradient
        # gradient = self._momentum_history
        # gradient = np.sign(gradient)
        min_, max_ = a.bounds()
        noise = (max_ - min_) * noise
        return noise

    def _run_one(self, *args, **kwargs):
        # reset momentum history every time we restart
        # gradient descent
        self._gradient_history = 0
        return super(RMSIterativeAttack, self)._run_one(*args, **kwargs)

    @call_decorator
    def __call__(self, input_or_adv, model_path=None, label=None, unpack=True,
                 binary_search=True,
                 epsilon=0.3,
                 stepsize=0.06,
                 iterations=10,
                 decay_factor=1.0,
                 gamma = 0.4,
                 random_start=False,
                 return_early=True):

        """Momentum-based iterative gradient attack known as
        Momentum Iterative Method.
        Parameters
        ----------
        input_or_adv : `numpy.ndarray` or :class:`Adversarial`
            The original, unperturbed input as a `numpy.ndarray` or
            an :class:`Adversarial` instance.
        label : int
            The reference label of the original input. Must be passed
            if `a` is a `numpy.ndarray`, must not be passed if `a` is
            an :class:`Adversarial` instance.
        unpack : bool
            If true, returns the adversarial input, otherwise returns
            the Adversarial object.
        binary_search : bool
            Whether to perform a binary search over epsilon and stepsize,
            keeping their ratio constant and using their values to start
            the search. If False, hyperparameters are not optimized.
            Can also be an integer, specifying the number of binary
            search steps (default 20).
        epsilon : float
            Limit on the perturbation size; if binary_search is True,
            this value is only for initialization and automatically
            adapted.
        stepsize : float
            Step size for gradient descent; if binary_search is True,
            this value is only for initialization and automatically
            adapted.
        iterations : int
            Number of iterations for each gradient descent run.
        decay_factor : float
            Decay factor used by the momentum term.
        random_start : bool
            Start the attack from a random point rather than from the
            original input.
        return_early : bool
            Whether an individual gradient descent run should stop as
            soon as an adversarial is found.
        """
        a = input_or_adv
        del input_or_adv
        del label
        del unpack

        assert epsilon > 0

        self._decay_factor = decay_factor
        self._gamma = gamma
        self._alpha = epsilon / 12.0
        # self._initial_temperature = _initial_temperature
        original = a.original_image.copy()
        positive_salience = None
        if model_path:
            positive_salience = find_salience(model_path, original)

        self._run(a, positive_salience, binary_search,
                  epsilon, stepsize, iterations,
                  random_start, return_early)

class AdamIterativeAttack(
        L2ClippingMixin,
        L2DistanceCheckMixin,
        IterativeProjectedGradientBaseAttack):
    def _gradient(self, a, x, class_, strict=True):
        # get current gradient
        noise = a.gradient(x, class_, strict=strict)
        noise = noise / max(1e-12, np.mean(np.abs(noise)))
        momentum = self._beta1 * self._gradient_history + (1 - self._beta1) * noise
        loss = self._beta2 * self._squared_gradient_history + (1 - self._beta2) * noise * noise
        self._gradient_history  = momentum
        self._squared_gradient_history = loss
        noise = self._alpha * (momentum/(np.sqrt(loss) + self._correction))
        min_, max_ = a.bounds()
        noise = (max_ - min_) * noise
        return noise

    def _run_one(self, *args, **kwargs):
        # reset momentum history every time we restart
        # gradient descent
        self._gradient_history = 0
        self._squared_gradient_history = 0
        return super(AdamIterativeAttack, self)._run_one(*args, **kwargs)

    @call_decorator
    def __call__(self, input_or_adv, model_path=None,label=None, unpack=True,
                 binary_search=True,
                 epsilon=0.3,
                 stepsize=0.06,
                 iterations=10,
                 decay_factor=1.0,
                 beta1 = 0.4,
                 beta2 = 0.4,
                 correction = 0.00001,
                 random_start=False,
                 return_early=True):

        """Momentum-based iterative gradient attack known as
        Momentum Iterative Method.
        Parameters
        ----------
        input_or_adv : `numpy.ndarray` or :class:`Adversarial`
            The original, unperturbed input as a `numpy.ndarray` or
            an :class:`Adversarial` instance.
        label : int
            The reference label of the original input. Must be passed
            if `a` is a `numpy.ndarray`, must not be passed if `a` is
            an :class:`Adversarial` instance.
        unpack : bool
            If true, returns the adversarial input, otherwise returns
            the Adversarial object.
        binary_search : bool
            Whether to perform a binary search over epsilon and stepsize,
            keeping their ratio constant and using their values to start
            the search. If False, hyperparameters are not optimized.
            Can also be an integer, specifying the number of binary
            search steps (default 20).
        epsilon : float
            Limit on the perturbation size; if binary_search is True,
            this value is only for initialization and automatically
            adapted.
        stepsize : float
            Step size for gradient descent; if binary_search is True,
            this value is only for initialization and automatically
            adapted.
        iterations : int
            Number of iterations for each gradient descent run.
        decay_factor : float
            Decay factor used by the momentum term.
        random_start : bool
            Start the attack from a random point rather than from the
            original input.
        return_early : bool
            Whether an individual gradient descent run should stop as
            soon as an adversarial is found.
        """
        a = input_or_adv
        del input_or_adv
        del label
        del unpack

        assert epsilon > 0

        self._decay_factor = decay_factor
        self._beta1 = beta1
        self._beta2 = beta2
        self._alpha = epsilon / 12.0
        self._correction = correction

        original = a.original_image.copy()
        positive_salience = None
        if model_path:
            positive_salience = find_salience(model_path, original)

        self._run(a, positive_salience, binary_search,
                  epsilon, stepsize, iterations,
                  random_start, return_early)

class AdagradIterativeAttack(
        L2ClippingMixin,
        L2DistanceCheckMixin,
        IterativeProjectedGradientBaseAttack):
    def _gradient(self, a, x, class_, strict=True):
        # get current gradient
        gradient = a.gradient(x, class_, strict=strict)
        noise = gradient / max(1e-12, np.mean(np.abs(gradient)))
        noise = noise * noise
        if self._gradient_history is None:
            self._gradient_history = noise
        else:
            assert self._gradient_history.shape == noise.shape
            self._gradient_history = self._gradient_history + noise
        noise = self._alpha/np.sqrt(self._gradient_history) * gradient
        # gradient = self._momentum_history
        # gradient = np.sign(gradient)
        min_, max_ = a.bounds()
        noise = (max_ - min_) * noise
        return noise

    def _run_one(self, *args, **kwargs):
        # reset momentum history every time we restart
        # gradient descent
        self._gradient_history = None
        return super(AdagradIterativeAttack, self)._run_one(*args, **kwargs)

    @call_decorator
    def __call__(self, input_or_adv, model_path = None, label=None, unpack=True,
                 binary_search=True,
                 epsilon=0.3,
                 stepsize=0.06,
                 iterations=10,
                 decay_factor=1.0,
                 random_start=False,
                 return_early=True):

        """Momentum-based iterative gradient attack known as
        Momentum Iterative Method.
        Parameters
        ----------
        input_or_adv : `numpy.ndarray` or :class:`Adversarial`
            The original, unperturbed input as a `numpy.ndarray` or
            an :class:`Adversarial` instance.
        label : int
            The reference label of the original input. Must be passed
            if `a` is a `numpy.ndarray`, must not be passed if `a` is
            an :class:`Adversarial` instance.
        unpack : bool
            If true, returns the adversarial input, otherwise returns
            the Adversarial object.
        binary_search : bool
            Whether to perform a binary search over epsilon and stepsize,
            keeping their ratio constant and using their values to start
            the search. If False, hyperparameters are not optimized.
            Can also be an integer, specifying the number of binary
            search steps (default 20).
        epsilon : float
            Limit on the perturbation size; if binary_search is True,
            this value is only for initialization and automatically
            adapted.
        stepsize : float
            Step size for gradient descent; if binary_search is True,
            this value is only for initialization and automatically
            adapted.
        iterations : int
            Number of iterations for each gradient descent run.
        decay_factor : float
            Decay factor used by the momentum term.
        random_start : bool
            Start the attack from a random point rather than from the
            original input.
        return_early : bool
            Whether an individual gradient descent run should stop as
            soon as an adversarial is found.
        """
        a = input_or_adv
        del input_or_adv
        del label
        del unpack

        assert epsilon > 0

        self._decay_factor = decay_factor
        self._alpha = epsilon / 12.0
        # self._initial_temperature = _initial_temperature

        self._run(a, None, binary_search,
                  epsilon, stepsize, iterations,
                  random_start, return_early)
