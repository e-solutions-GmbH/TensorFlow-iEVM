import numpy as np
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import LabelEncoder


class IncrementalLabelEncoder(LabelEncoder):
    """
    An addon of the scikit-learn LabelEncoder enabling partial fits.

    Attributes
    ----------
    classes_ :  ndarray of shape (n_classes,)
                Holds the label for each class.
    """

    def __init__(self):
        """
        Initialize instance.
        """
        self.classes_ = None

    def partial_fit(self, y: np.ndarray):
        """
        Updates the label encoder.

        Note that the internal class representation will be sorted, which may invalidate saved labels in a classifier.
        A workaround to this problem is to use inverse_transform for the stored labels and then transform back, e.g.:

        clf = BaseEstimator()
        le = IncrementalLabelEncoder()

        # batch 1
        yt1 = le.fit_transform(y1)
        clf.fit(x, yt1)

        # batch 2
        cfl_ity = le.inverse_transform(clf.y)
        yt2 = le.partial_fit_transform(y2)
        clf.y = le.transform(cfl_ity)
        clf.partial_fit(x2, yt2)

        Parameters
        ----------
        y : array-like of shape (n_samples,). Target values.

        Returns
        -------
        self : returns an instance of self. Fitted label encoder.
        """
        if self.is_empty():
            return self.fit(y)

        return self.fit(np.concatenate([y, self.classes_], axis=0))

    def transform(self, y: np.ndarray):
        """
        Transform labels to normalized encoding.

        Parameters
        ----------
        y : array-like of shape (n_samples,). Target values.

        Returns
        -------
        y : array-like of shape (n_samples,)
            Labels as normalized encodings.
        """
        if self.is_empty():
            raise NotFittedError("{} is not fitted yet.".format(self.__class__.__name__))

        return super(IncrementalLabelEncoder, self).transform(y)

    def partial_fit_transform(self, y: np.ndarray):
        """
        Update label encoder and return encoded labels.

        Note that the internal class representation will be sorted, which may invalidate saved labels in a classifier.
        For a workaround see 'partial_fit'.

        Parameters
        ----------
        y : array-like of shape (n_samples,). Target values.

        Returns
        -------
        y : array-like of shape (n_samples,). Encoded labels.
        """

        self.partial_fit(y)
        return super(IncrementalLabelEncoder, self).transform(y)

    def inverse_transform(self, y) -> np.ndarray:
        """
        Transform labels back to original encoding.

        Parameters
        ----------
        y : ndarray of shape (n_samples,). Target values.

        Returns
        -------
        y : ndarray of shape (n_samples,). Original encoding.
        """

        if self.is_empty():
            raise NotFittedError("{} is not fitted yet.".format(self.__class__.__name__))

        return super(IncrementalLabelEncoder, self).inverse_transform(y)

    def reset(self):
        """
        Resets the label encoder.
        """
        self.classes_ = None

    def is_empty(self) -> bool:
        """
        Check if instance is fitted.

        Returns
        -------
        True if the instance is not fitted.
        """
        return self.classes_ is None or len(self.classes_) == 0
