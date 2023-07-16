#################################
# Your name:Guy Shemesh
#################################

import numpy as np
import matplotlib.pyplot as plt
import intervals


class Assignment2(object):
    """Assignment 2 skeleton.

    Please use these function signatures for this assignment and submit this file, together with the intervals.py.
    """

    def sample_from_D(self, m):
        """Sample m data samples from D.
        Input: m - an integer, the size of the data sample.

        Returns: np.ndarray of shape (m,2) :
                A two dimensional array of size m that contains the pairs where drawn from the distribution P.
        """
        X = np.random.uniform(0, 1, m)
        X.sort()
        Y = np.array([np.random.choice([0, 1], p=self.probability_calculate(x)) for x in X])
        ret= np.column_stack((X, Y))
        return ret



    def experiment_m_range_erm(self, m_first, m_last, step, k, T):
        """Runs the ERM algorithm.
        Calculates the empirical error and the true error.
        Plots the average empirical and true errors.
        Input: m_first - an integer, the smallest size of the data sample in the range.
               m_last - an integer, the largest size of the data sample in the range.
               step - an integer, the difference between the size of m in each loop.
               k - an integer, the maximum number of intervals.
               T - an integer, the number of times the experiment is performed.

        Returns: np.ndarray of shape (n_steps,2).
            A two dimensional array that contains the average empirical error
            and the average true error for each m in the range accordingly.
        """
        avg_errors_arr = []
        for n in range(m_first, m_last + 1, step):
            emp_and_true_errors = [self.emp_true_errors_for_experiment(n, k) for t in range(T)]
            avg_emp_true_errors = [sum(error) / T for error in zip(*emp_and_true_errors)]
            avg_errors_arr.append(avg_emp_true_errors)
        avg_errors = np.asarray(avg_errors_arr)
        n= np.arange(m_first, m_last + 1, step)
        plt.xlabel("n")
        plt.plot(n, avg_errors[:, 0], label="empirical error")
        plt.plot(n, avg_errors[:, 1], label="true error")
        plt.legend()
        plt.show()
        return avg_errors

    def experiment_k_range_erm(self, m, k_first, k_last, step):
        """Finds the best hypothesis for k= 1,2,...,10.
        Plots the empirical and true errors as a function of k.
        Input: m - an integer, the size of the data sample.
               k_first - an integer, the maximum number of intervals in the first experiment.
               m_last - an integer, the maximum number of intervals in the last experiment.
               step - an integer, the difference between the size of k in each experiment.

        Returns: The best k value (an integer) according to the ERM algorithm.
        """
        sample = self.sample_from_D(m)
        array_of_errors = np.array([self.emp_true_errors_sample(sample, m, k) for k in range(k_first, k_last + 1, step)])
        k_min = np.argmin(array_of_errors[:, 0]) * step + k_first
        k = np.arange(k_first, k_last + 1, step)
        plt.xlabel("k")
        plt.plot(k, array_of_errors[:, 0], label="empirical error")
        plt.plot(k, array_of_errors[:, 1], label="true error")
        plt.legend()
        plt.show()
        return k_min


    def cross_validation(self, m):
        """Finds a k that gives a good test error.
        Input: m - an integer, the size of the data sample.

        Returns: The best k value (an integer) found by the cross validation algorithm.
        """
        arr= self.sample_from_D(m)
        np.random.shuffle(arr)
        train, validation = arr[:int(m * 0.8)], arr[int(m * 0.8):]
        train = train[train[:, 0].argsort()]
        best_hypo = [intervals.find_best_interval(train[:, 0], train[:, 1], k)[0] for k in range(1, 11)]
        validation_errors = np.array([self.emp_errors(validation, intervals_list) for intervals_list in best_hypo])
        best_k= np.argmin(validation_errors) + 1
        best_hypo_validation_error=validation_errors[best_k-1]
        k = [i for i in range(1,11)]
        plt.xlabel("k")
        plt.plot(k,validation_errors, label="validation error")
        plt.legend()
        plt.show()
        interval,s=intervals.find_best_interval(train[:, 0], train[:, 1], best_k)
        print("the best k for the best validation error :")
        print(best_k)
        print("the best validation error for the best k is :")
        print(best_hypo_validation_error)
        print("the intervals are :")
        for i in range(0,best_k):
         print(interval[i])
        return best_k
    #################################
    # Place for additional methods
    def probability_calculate(self,x):
        if ((0 <= x and x <= 0.2) or (0.4 <= x and x <= 0.6) or (0.8 <= x and x <= 1)):
            return [0.2, 0.8]
        return [0.9, 0.1]

    def emp_true_errors_for_experiment(self, n, k):
     sample = self.sample_from_D(n)
     true_error, emp_error = intervals.find_best_interval(sample[:, 0], sample[:, 1], k)
     emp_error = emp_error / n
     true_error = self.true_errors(true_error)
     return emp_error, true_error

    def emp_true_errors_sample(self, sample, n, k):
      true_error, emp_error = intervals.find_best_interval(sample[:, 0], sample[:, 1], k)
      emp_error = emp_error / n
      true_error=self.true_errors(true_error)
      return emp_error, true_error

    def true_errors(self,intervals):
     intervals_Y_1_high_p = [(0, 0.2), (0.4, 0.6), (0.8, 1)]
     intervals_Y_1_low_p = [(0.2, 0.4), (0.6, 0.8)]
     len_cur_and_high_p = self.len_intersection_two_lists(intervals,intervals_Y_1_high_p)
     len_cur_and_low_p = self.len_intersection_two_lists(intervals,intervals_Y_1_low_p)
     len_not_cur_and_high_p = 0.6 - len_cur_and_high_p
     len_not_cur_and_low_p = 0.4 - len_cur_and_low_p
     return 0.8 * len_not_cur_and_high_p + 0.1 * len_not_cur_and_low_p +0.2 * len_cur_and_high_p + 0.9 * len_cur_and_low_p

    def len_intersection_two_lists(self,list1,list2):
     index_list1 = 0
     index_list2 = 0
     len_intersection_two_lists = 0
     while ((index_list1 < len(list1)) and (index_list2 < len(list2))):
         a = max(list1[index_list1][0], list2[index_list2][0])
         b = min(list1[index_list1][1], list2[index_list2][1])
         if a < b:
             len_intersection_two_lists += (b - a)
         if list1[index_list1][1] == list2[index_list2][1]:
             index_list1 += 1
             index_list2 += 1
         elif list1[index_list1][1] < list2[index_list2][1]:
             index_list1 += 1
         else:
             index_list2 += 1
     return len_intersection_two_lists

    def emp_errors(self,arr,inter_list):
      return sum([self.L0_1(inter_list, x, y) for x, y in arr]) / len(arr)

    def L0_1(self, inter_list, x, y):
        x_in_interval = self.check_if_in_the_interval(inter_list, x)
        if (x_in_interval and y == 1) or (not x_in_interval and y == 0):
            return 0
        return 1

    def check_if_in_the_interval(self,inter_list,x):
     for interval in inter_list:
        if (interval[0] <= x)  and (x <= interval[1]):
            return True
     return False

   #################################
if __name__ == '__main__':
    ass = Assignment2()
    ass.experiment_m_range_erm(10, 100, 5, 3, 100)
    ass.experiment_k_range_erm(1500, 1, 10, 1)
    ass.cross_validation(1500)

