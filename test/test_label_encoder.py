import unittest

import numpy as np
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import LabelEncoder

from tf_ievm.preprocessing import IncrementalLabelEncoder
from tf_ievm.util import batch_generator


class TestEVM(unittest.TestCase):

    def setUp(self) -> None:
        self.seed = 123
        np.random.seed(self.seed)

        # do not change order -> may invalidate tests
        self.y_list = [
            np.arange(5),  # sorted arange-like labels
            np.arange(-1, 5),  # sorted arange-like labels
            np.arange(5, -2, step=-1),  # inverse sorted arange-like labels
            np.random.randint(-500, 500, (250,))  # random labels
        ]

        self.dtype_list = [int, float, str]
        self.unknown_label_list = [-1, 0, 5]
        self.batch_size_list = [1, 3, 100]

    def test_init(self):
        enc = IncrementalLabelEncoder()
        self.assertTrue(enc.is_empty())

    def test_empty(self):
        enc = IncrementalLabelEncoder()
        self.assertTrue(enc.is_empty())
        self.assertRaises(NotFittedError, enc.transform, [0])
        self.assertRaises(NotFittedError, enc.inverse_transform, [0])

    def test_empty_labels(self):
        enc = IncrementalLabelEncoder()
        y = np.array([0])
        y_empty = np.array([])

        # fit -> transform -> inverse_transform
        enc.fit(y)
        yt = enc.transform(y_empty)
        np.testing.assert_equal(yt, y_empty)
        yit = enc.inverse_transform(yt)
        np.testing.assert_equal(yit, y_empty)

        # fit_transform
        enc.reset()
        yt = enc.fit_transform(y_empty)
        np.testing.assert_equal(yt, y_empty)

        # partial_fit -> transform
        enc.reset()
        enc.partial_fit(y)
        yt = enc.transform(y_empty)
        np.testing.assert_equal(yt, y_empty)

        # partial_fit_transform
        enc.reset()
        yt = enc.partial_fit_transform(y_empty)
        np.testing.assert_equal(yt, y_empty)

    def test_fit_and_transform(self):
        sk_enc = LabelEncoder()
        i_enc = IncrementalLabelEncoder()

        for i, y in enumerate(self.y_list):
            for dtype in self.dtype_list:
                y_typed = y.astype(dtype)

                # to check that the label encoder does not modify the original labels
                y_typed_copy = y_typed.copy()

                # fit
                sk_enc.fit(y_typed)
                np.testing.assert_equal(y_typed, y_typed_copy)

                i_enc.fit(y_typed)
                np.testing.assert_equal(y_typed, y_typed_copy)
                self.assertFalse(i_enc.is_empty())

                # transform
                sk_y_t = sk_enc.transform(y_typed)
                np.testing.assert_equal(y_typed, y_typed_copy)

                i_y_t = i_enc.transform(y_typed)
                np.testing.assert_equal(y_typed, y_typed_copy)

                # arange-like labels should not change
                if i == 0:
                    np.testing.assert_equal(sk_y_t, y)
                elif i == 1:
                    np.testing.assert_equal(sk_y_t, np.arange(len(y)))
                elif i == 2:
                    np.testing.assert_equal(sk_y_t, np.arange(len(y) - 1, -1, step=-1))

                # check if sklearn and the addon do the same
                np.testing.assert_equal(sk_y_t, i_y_t)

                # unknown classes
                if dtype == str:
                    self.assertRaises(ValueError, sk_enc.transform, ["foo"])
                    self.assertRaises(ValueError, sk_enc.inverse_transform, [len(sk_enc.classes_) + 1])
                    self.assertRaises(ValueError, i_enc.transform, ["foo"])
                    self.assertRaises(ValueError, i_enc.inverse_transform, [len(i_enc.classes_) + 1])
                else:
                    self.assertRaises(ValueError, sk_enc.transform, [np.min(y_typed) - 1])
                    self.assertRaises(ValueError, sk_enc.inverse_transform, [np.min(sk_enc.classes_) - 1])
                    self.assertRaises(ValueError, i_enc.transform, [np.min(y_typed) - 1])
                    self.assertRaises(ValueError, i_enc.inverse_transform, [np.min(i_enc.classes_) - 1])

                # inverse transform
                sk_y_t_copy = sk_y_t.copy()
                sk_y_it = sk_enc.inverse_transform(sk_y_t)
                np.testing.assert_equal(sk_y_t, sk_y_t_copy)
                np.testing.assert_equal(sk_y_it, y_typed)

                i_y_t_copy = i_y_t.copy()
                i_y_it = i_enc.inverse_transform(i_y_t)
                np.testing.assert_equal(i_y_t, i_y_t_copy)
                np.testing.assert_equal(i_y_it, y_typed)

                # reset -> empty
                i_enc.reset()
                self.assertTrue(i_enc.is_empty())

    def test_fit_transform(self):
        sk_enc = LabelEncoder()
        i_enc = IncrementalLabelEncoder()

        for i, y in enumerate(self.y_list):
            for dtype in self.dtype_list:
                y_typed = y.astype(dtype)

                # to check that no transform modifies the original array
                y_typed_copy = y_typed.copy()

                # fit_transform
                sk_y_t = sk_enc.fit_transform(y_typed)
                np.testing.assert_equal(y_typed, y_typed_copy)

                i_y_t = i_enc.fit_transform(y_typed)
                np.testing.assert_equal(y_typed, y_typed_copy)
                self.assertFalse(i_enc.is_empty())

                # arange-like labels should not change
                if i == 0:
                    np.testing.assert_equal(sk_y_t, y)
                elif i == 1:
                    np.testing.assert_equal(sk_y_t, np.arange(len(y)))
                elif i == 2:
                    np.testing.assert_equal(sk_y_t, np.arange(len(y) - 1, -1, step=-1))

                # check if scikit-learn and the addon do the same
                np.testing.assert_equal(sk_y_t, i_y_t)

                # unknown classes
                if dtype == str:
                    self.assertRaises(ValueError, sk_enc.transform, ["foo"])
                    self.assertRaises(ValueError, sk_enc.inverse_transform, [len(sk_enc.classes_) + 1])
                    self.assertRaises(ValueError, i_enc.transform, ["foo"])
                    self.assertRaises(ValueError, i_enc.inverse_transform, [len(i_enc.classes_) + 1])
                else:
                    self.assertRaises(ValueError, sk_enc.transform, [np.min(y_typed) - 1])
                    self.assertRaises(ValueError, sk_enc.inverse_transform, [np.min(sk_enc.classes_) - 1])
                    self.assertRaises(ValueError, i_enc.transform, [np.min(y_typed) - 1])
                    self.assertRaises(ValueError, i_enc.inverse_transform, [np.min(i_enc.classes_) - 1])

                # inverse transform
                sk_y_t_copy = sk_y_t.copy()
                sk_y_it = sk_enc.inverse_transform(sk_y_t)
                np.testing.assert_equal(sk_y_t, sk_y_t_copy)
                np.testing.assert_equal(sk_y_it, y_typed)

                i_y_t_copy = i_y_t.copy()
                i_y_it = i_enc.inverse_transform(i_y_t)
                np.testing.assert_equal(i_y_t, i_y_t_copy)
                np.testing.assert_equal(i_y_it, y_typed)

                # reset -> empty
                i_enc.reset()
                self.assertTrue(i_enc.is_empty())

    def test_partial_fit_and_transform(self):
        sk_enc = LabelEncoder()
        for i, y in enumerate(self.y_list):
            for dtype in self.dtype_list:
                y_typed = y.astype(dtype)
                for batch_size in self.batch_size_list:
                    i_enc = IncrementalLabelEncoder()
                    y_seen = []
                    y_typed_seen = []
                    for [cy, cy_typed] in batch_generator(y, y_typed, batch_size=batch_size):
                        y_seen.append(cy)
                        y_seen_concat = np.concatenate(y_seen, axis=0)
                        y_typed_seen.append(cy_typed)
                        y_typed_seen_concat = np.concatenate(y_typed_seen, axis=0)

                        # sklearn fit all seen
                        y_typed_seen_concat_copy = y_typed_seen_concat.copy()
                        sk_enc.fit(y_typed_seen_concat)
                        np.testing.assert_equal(y_typed_seen_concat, y_typed_seen_concat_copy)

                        # addon partial_fit batch
                        cy_typed_copy = cy_typed.copy()
                        i_enc.partial_fit(cy_typed)
                        np.testing.assert_equal(cy_typed, cy_typed_copy)
                        self.assertFalse(i_enc.is_empty())

                        # transform batch
                        sk_cy_t = sk_enc.transform(cy_typed)
                        np.testing.assert_equal(cy_typed, cy_typed_copy)
                        i_cy_t = i_enc.transform(cy_typed)
                        np.testing.assert_equal(cy_typed, cy_typed_copy)

                        # transform all seen
                        sk_y_seen_concat_t = sk_enc.transform(y_typed_seen_concat)
                        np.testing.assert_equal(y_typed_seen_concat, y_typed_seen_concat_copy)
                        i_y_seen_concat_t = i_enc.transform(y_typed_seen_concat)
                        np.testing.assert_equal(y_typed_seen_concat, y_typed_seen_concat_copy)

                        # arange-like labels should not change
                        if i == 0:
                            np.testing.assert_equal(sk_cy_t, cy)
                            np.testing.assert_equal(sk_y_seen_concat_t, y_seen_concat)
                        elif i == 1:
                            np.testing.assert_equal(sk_y_seen_concat_t, np.arange(len(y_seen_concat)))
                        elif i == 2:
                            np.testing.assert_equal(sk_y_seen_concat_t, np.arange(len(y_seen_concat) - 1, -1, step=-1))

                        # check if sklearn and the addon do the same
                        np.testing.assert_equal(sk_cy_t, i_cy_t)
                        np.testing.assert_equal(sk_y_seen_concat_t, i_y_seen_concat_t)

                        # unknown classes
                        if dtype == str:
                            self.assertRaises(ValueError, sk_enc.transform, ["foo"])
                            self.assertRaises(ValueError, sk_enc.inverse_transform, [len(sk_enc.classes_) + 1])
                            self.assertRaises(ValueError, i_enc.transform, ["foo"])
                            self.assertRaises(ValueError, i_enc.inverse_transform, [len(i_enc.classes_) + 1])
                        else:
                            self.assertRaises(ValueError, sk_enc.transform, [np.min(y_typed) - 1])
                            self.assertRaises(ValueError, sk_enc.inverse_transform, [len(sk_enc.classes_) + 1])
                            self.assertRaises(ValueError, i_enc.transform, [np.min(y_typed) - 1])
                            self.assertRaises(ValueError, i_enc.inverse_transform, [len(i_enc.classes_) + 1])

                        # inverse transform batch
                        sk_cy_t_copy = sk_cy_t.copy()
                        sk_cy_it = sk_enc.inverse_transform(sk_cy_t)
                        np.testing.assert_equal(sk_cy_t, sk_cy_t_copy)
                        np.testing.assert_equal(sk_cy_it, cy_typed)

                        i_cy_t_copy = i_cy_t.copy()
                        i_cy_it = i_enc.inverse_transform(i_cy_t)
                        np.testing.assert_equal(i_cy_t, i_cy_t_copy)
                        np.testing.assert_equal(i_cy_it, cy_typed)

                        # inverse transform all seen
                        sk_y_seen_concat_t_copy = sk_y_seen_concat_t.copy()
                        sk_y_seen_it = sk_enc.inverse_transform(sk_y_seen_concat_t)
                        np.testing.assert_equal(sk_y_seen_concat_t, sk_y_seen_concat_t_copy)
                        np.testing.assert_equal(sk_y_seen_it, y_typed_seen_concat)

                        i_y_seen_concat_t_copy = i_y_seen_concat_t.copy()
                        i_y_seen_it = i_enc.inverse_transform(i_y_seen_concat_t)
                        np.testing.assert_equal(i_y_seen_concat_t, i_y_seen_concat_t_copy)
                        np.testing.assert_equal(i_y_seen_it, y_typed_seen_concat)

    def test_partial_fit_transform(self):
        sk_enc = LabelEncoder()

        y1 = np.arange(5)  # sorted arange-like labels
        y2 = np.random.randint(-500, 500, (250,))  # random labels
        for i, y in enumerate([y1, y2]):
            for dtype in [int, float, str]:
                y_typed = y.astype(dtype)
                for batch_size in [1, 3, 100]:
                    i_enc = IncrementalLabelEncoder()
                    y_seen = []
                    y_typed_seen = []
                    for [cy, cy_typed] in batch_generator(y, y_typed, batch_size=batch_size):
                        y_seen.append(cy)
                        y_seen_concat = np.concatenate(y_seen, axis=0)
                        y_typed_seen.append(cy_typed)
                        y_typed_seen_concat = np.concatenate(y_typed_seen, axis=0)

                        # sklearn fit_transform all seen
                        y_typed_seen_concat_copy = y_typed_seen_concat.copy()
                        sk_y_seen_concat_t = sk_enc.fit_transform(y_typed_seen_concat)
                        np.testing.assert_equal(y_typed_seen_concat, y_typed_seen_concat_copy)

                        # sklearn transform batch
                        cy_typed_copy = cy_typed.copy()
                        sk_cy_t = sk_enc.transform(cy_typed)
                        np.testing.assert_equal(cy_typed, cy_typed_copy)

                        # addon partial_fit_transform batch
                        i_cy_t = i_enc.partial_fit_transform(cy_typed)
                        np.testing.assert_equal(cy_typed, cy_typed_copy)
                        self.assertFalse(i_enc.is_empty())

                        # addon transform all seen
                        i_y_seen_concat_t = i_enc.transform(y_typed_seen_concat)
                        np.testing.assert_equal(y_typed_seen_concat, y_typed_seen_concat_copy)

                        # arange-like labels should not change
                        if i == 0:
                            np.testing.assert_equal(sk_cy_t, cy)
                            np.testing.assert_equal(sk_y_seen_concat_t, y_seen_concat)

                        # check if sklearn and the addon do the same
                        np.testing.assert_equal(sk_cy_t, i_cy_t)
                        np.testing.assert_equal(sk_y_seen_concat_t, i_y_seen_concat_t)

                        # unknown classes
                        if dtype == str:
                            self.assertRaises(ValueError, sk_enc.transform, ["foo"])
                            self.assertRaises(ValueError, sk_enc.inverse_transform, [len(sk_enc.classes_) + 1])
                            self.assertRaises(ValueError, i_enc.transform, ["foo"])
                            self.assertRaises(ValueError, i_enc.inverse_transform, [len(i_enc.classes_) + 1])
                        else:
                            self.assertRaises(ValueError, sk_enc.transform, [np.min(y_typed) - 1])
                            self.assertRaises(ValueError, sk_enc.inverse_transform, [np.min(sk_enc.classes_) - 1])
                            self.assertRaises(ValueError, i_enc.transform, [np.min(y_typed) - 1])
                            self.assertRaises(ValueError, i_enc.inverse_transform, [np.min(i_enc.classes_) - 1])

                        # inverse transform batch
                        sk_cy_t_copy = sk_cy_t.copy()
                        sk_cy_it = sk_enc.inverse_transform(sk_cy_t)
                        np.testing.assert_equal(sk_cy_t, sk_cy_t_copy)
                        np.testing.assert_equal(sk_cy_it, cy_typed)

                        i_cy_t_copy = i_cy_t.copy()
                        i_cy_it = i_enc.inverse_transform(i_cy_t)
                        np.testing.assert_equal(i_cy_t, i_cy_t_copy)
                        np.testing.assert_equal(i_cy_it, cy_typed)

                        # inverse transform all seen
                        sk_y_seen_concat_t_copy = sk_y_seen_concat_t.copy()
                        sk_y_seen_it = sk_enc.inverse_transform(sk_y_seen_concat_t)
                        np.testing.assert_equal(sk_y_seen_concat_t, sk_y_seen_concat_t_copy)
                        np.testing.assert_equal(sk_y_seen_it, y_typed_seen_concat)

                        i_y_seen_concat_t_copy = i_y_seen_concat_t.copy()
                        i_y_seen_it = i_enc.inverse_transform(i_y_seen_concat_t)
                        np.testing.assert_equal(i_y_seen_concat_t, i_y_seen_concat_t_copy)
                        np.testing.assert_equal(i_y_seen_it, y_typed_seen_concat)


if __name__ == '__main__':
    unittest.main()
