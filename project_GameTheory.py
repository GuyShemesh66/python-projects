import random
from typing import List, Dict, Tuple, Set, Optional
import numpy as np
import matplotlib.pyplot as plt

# Constants
NUM_COLLEGES = 10
NUM_STUDENTS = 1000
# College quota constants - to make the data more realistic, we add normal distribution noise to the base 
# quota of each college, to simulate the fact that colleges have different quotas and the quotas are not 
# exactly known.
BASE_COLLEGE_QUOTA = 50
COLLEGE_QUOTA_NOISE_STD = 5
# Student preference constants - to make the data more realistic, we add normal distribution noise to the 
# base preference of each student, to simulate the fact that students have different preferences and the 
# preferences are not exactly known - e.g. not always all students would prefer the top college
STUDENT_PREF_NOISE_STD = 2
# Student score constants - to make the data more realistic, we add normal distribution noise to the 
# base score of each student, to simulate the fact that students have different scores and the scores 
# are not exactly known.
STUDENT_SCORE_MEAN = 700
STUDENT_SCORE_STD = 100
RANDOM_SEED = 42
NUM_GROUPS = 4
# When we divide into weighted groupd - i.e. put the students into groups based on their score,
# we add noise to the scores to make the groups more realistic - i.e. not all students in the 
# top group are the highest scoring students.
WEIGHTED_GROUPS_NOISE_STD = 25

def set_random_seed() -> None:
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

def generate_colleges_and_students() -> Tuple[List[str], List[str]]:
    colleges = [f"College#{i+1}" for i in range(NUM_COLLEGES)]
    students = [f"Student#{i+1}" for i in range(NUM_STUDENTS)]
    return colleges, students

def generate_college_quotas(colleges: List[str]) -> Dict[str, int]:
    return {college: max(1, int(np.random.normal(BASE_COLLEGE_QUOTA, COLLEGE_QUOTA_NOISE_STD))) for college in colleges}

def generate_student_scores(students: List[str]) -> Dict[str, int]:
    return {student: max(0, int(np.random.normal(STUDENT_SCORE_MEAN, STUDENT_SCORE_STD))) for student in students}

def generate_student_preferences(colleges: List[str]) -> List[str]:
    base_preferences = list(range(NUM_COLLEGES))
    noise = np.random.normal(0, STUDENT_PREF_NOISE_STD, NUM_COLLEGES)
    noisy_preferences = [x + noise[i] for i, x in enumerate(base_preferences)]
    return [colleges[i] for i in np.argsort(noisy_preferences)]

def generate_all_student_preferences(students: List[str], colleges: List[str]) -> Dict[str, List[str]]:
    return {student: generate_student_preferences(colleges) for student in students}

def generate_college_preferences(students: List[str], student_scores: Dict[str, int]) -> List[str]:
    return sorted(students, key=lambda s: student_scores[s], reverse=True)

def generate_all_college_preferences(colleges: List[str], students: List[str], student_scores: Dict[str, int]) -> Dict[str, List[str]]:
    return {college: generate_college_preferences(students, student_scores) for college in colleges}

def assign_student_groups(students: List[str]) -> Dict[str, int]:
    return {student: random.randint(1, NUM_GROUPS) for student in students}

def assign_score_based_groups(students: List[str], student_scores: Dict[str, int]) -> Dict[str, int]:
    sorted_students = sorted(students, key=lambda s: student_scores[s] + np.random.normal(0, WEIGHTED_GROUPS_NOISE_STD), reverse=True)
    group_size = len(students) // NUM_GROUPS
    return {student: (i // group_size) + 1 for i, student in enumerate(sorted_students)}

def generate_college_group_quotas(college_quotas: Dict[str, int]) -> Dict[str, Dict[int, int]]:
    return {college: {group: max(1, int(quota / NUM_GROUPS)) 
                      for group in range(1, NUM_GROUPS + 1)}
            for college, quota in college_quotas.items()}

def initialize_matches(students: List[str], colleges: List[str], quotas: Dict[str, int]) -> Tuple[Set[str], Dict[str, int], Dict[str, Optional[str]], Dict[str, List[str]]]:
    unmatched_students = set(students)
    college_slots = quotas.copy()
    student_matches = {student: None for student in students}
    college_matches = {college: [] for college in colleges}
    return unmatched_students, college_slots, student_matches, college_matches

def process_student_application(
    student: str,
    college: str,
    college_prefs: Dict[str, List[str]],
    college_slots: Dict[str, int],
    student_matches: Dict[str, Optional[str]],
    college_matches: Dict[str, List[str]],
    student_groups: Optional[Dict[str, int]] = None,
    college_group_quotas: Optional[Dict[str, Dict[int, int]]] = None
) -> Tuple[bool, List[str]]:
    current_students = college_matches[college] + [student]
    sorted_students = sorted(current_students, key=lambda s: college_prefs[college].index(s))
    accepted_students = sorted_students[:college_slots[college]]
    rejected_students = sorted_students[college_slots[college]:]
    
    college_matches[college] = accepted_students
    
    newly_rejected = []
    for rejected_student in rejected_students:
        if student_matches[rejected_student] == college:
            student_matches[rejected_student] = None
            newly_rejected.append(rejected_student)
    
    if student in accepted_students:
        student_matches[student] = college
        return True, newly_rejected
    else:
        return False, newly_rejected

def process_student_application_with_groups(
    student: str,
    college: str,
    college_prefs: Dict[str, List[str]],
    college_slots: Dict[str, int],
    student_matches: Dict[str, Optional[str]],
    college_matches: Dict[str, List[str]],
    student_groups: Dict[str, int],
    college_group_quotas: Dict[str, Dict[int, int]]
) -> Tuple[bool, List[str]]:
    """
    This method processes a student's application to a college, taking into account group quotas.
    It first separates the current college matches into students from the same group as the applicant
    and students from other groups. Then, it sorts the students from the applicant's group based on
    the college's preferences and compares this to the group quota. If there's space within the quota,
    the student is accepted. If not, the student is compared to the least preferred student in their
    group currently matched to the college. If the applicant is more preferred, they replace the least
    preferred student, who becomes unmatched. The method returns whether the student was accepted and
    a list of any students who became unmatched as a result.
    """
    student_group = student_groups[student]
    
    current_group_students = [s for s in college_matches[college] if student_groups[s] == student_group]
    students_not_in_this_group = [s for s in college_matches[college] if student_groups[s] != student_group]
    current_group_students.append(student)

    sorted_group_students = sorted(current_group_students, key=lambda s: college_prefs[college].index(s))
    group_quota = college_group_quotas[college][student_group]
    accepted_students = sorted_group_students[:group_quota] + students_not_in_this_group
    college_matches[college] = accepted_students
    if len(sorted_group_students) < group_quota:
        student_matches[student] = college
        return True, []
    rejected_student = sorted_group_students[-1]
    if rejected_student == student:
        return False, []
    
    student_matches[rejected_student] = None
    student_matches[student] = college
    return True, [rejected_student]

def gale_shapley(
    student_prefs: Dict[str, List[str]],
    college_prefs: Dict[str, List[str]],
    quotas: Dict[str, int],
    students: List[str],
    colleges: List[str],
    process_application: callable,
    student_groups: Optional[Dict[str, int]] = None,
    college_group_quotas: Optional[Dict[str, Dict[int, int]]] = None
) -> Dict[str, Optional[str]]:
    unmatched_students, college_slots, student_matches, college_matches = initialize_matches(students, colleges, quotas)

    iterations_without_new_matches = 0
    while unmatched_students and iterations_without_new_matches < NUM_COLLEGES:
        new_matches = False
        for student in list(unmatched_students):
            for college in student_prefs[student]:
                result, newly_rejected = process_application(
                    student, college, college_prefs, college_slots, 
                    student_matches, college_matches, student_groups, college_group_quotas
                )
                unmatched_students.update(newly_rejected)
                if result:
                    unmatched_students.remove(student)
                    new_matches = True
                    break
        if new_matches:
            iterations_without_new_matches = 0
        else:
            iterations_without_new_matches += 1

    return student_matches

def generate_average_student_score_graph(colleges: List[str], final_student_matches: Dict[str, Optional[str]], final_student_matches_with_random_groups: Dict[str, Optional[str]], final_student_matches_with_score_groups: Dict[str, Optional[str]], student_scores: Dict[str, int], filename: str) -> None:
    college_avg_scores = {}
    college_avg_scores_with_random_groups = {}
    college_avg_scores_with_score_groups = {}
    
    for college in colleges:
        matched_students = [s for s, c in final_student_matches.items() if c == college]
        matched_students_with_random_groups = [s for s, c in final_student_matches_with_random_groups.items() if c == college]
        matched_students_with_score_groups = [s for s, c in final_student_matches_with_score_groups.items() if c == college]
        
        if matched_students:
            avg_score = np.mean([student_scores[s] for s in matched_students])
            college_avg_scores[college] = avg_score
        
        if matched_students_with_random_groups:
            avg_score_with_random_groups = np.mean([student_scores[s] for s in matched_students_with_random_groups])
            college_avg_scores_with_random_groups[college] = avg_score_with_random_groups
        
        if matched_students_with_score_groups:
            avg_score_with_score_groups = np.mean([student_scores[s] for s in matched_students_with_score_groups])
            college_avg_scores_with_score_groups[college] = avg_score_with_score_groups
    
    plt.figure(figsize=(14, 6))
    x = range(len(colleges))
    width = 0.25
    
    plt.bar([i - width for i in x], [college_avg_scores.get(college, 0) for college in colleges], width, label='Without Groups', color='purple')
    plt.bar([i for i in x], [college_avg_scores_with_random_groups.get(college, 0) for college in colleges], width, label='With Random Groups', color='blue')
    plt.bar([i + width for i in x], [college_avg_scores_with_score_groups.get(college, 0) for college in colleges], width, label='With Score-based Groups', color='#800020')
    
    plt.title("Average Student Score per College")
    plt.xlabel("Colleges")
    plt.ylabel("Average Student Score")
    plt.xticks(x, colleges, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

#####################################################################
# These are only used to debug, and do not affect the result itself #
#####################################################################   

def preference_satisfaction(student: str, final_student_matches: Dict[str, Optional[str]], student_preferences: Dict[str, List[str]]) -> float:
    match = final_student_matches[student]
    if match is None:
        return 0
    return 1 - (student_preferences[student].index(match) / NUM_COLLEGES)

def print_college_quotas(college_quotas: Dict[str, int]) -> None:
    print("College Quotas and Minimum Scores:")
    for college, quota in college_quotas.items():
        print(f"{college}: Quota = {quota}")

def print_lowest_scoring_students(students: List[str], student_scores: Dict[str, int], final_student_matches: Dict[str, Optional[str]]) -> None:
    print("\nFinal Matches (10 students with lowest scores):")
    lowest_scoring_students = sorted(students, key=lambda s: student_scores[s])[:10]
    for student in lowest_scoring_students:
        print(f"{student} (Score: {student_scores[student]}) -> {final_student_matches[student]}")

def print_highest_scoring_students(students: List[str], student_scores: Dict[str, int], final_student_matches: Dict[str, Optional[str]]) -> None:
    print("\nFinal Matches (10 students with highest scores):")
    highest_scoring_students = sorted(students, key=lambda s: student_scores[s], reverse=True)[:10]
    for student in highest_scoring_students:
        print(f"{student} (Score: {student_scores[student]}) -> {final_student_matches[student]}")

def print_college_fill_rates(colleges: List[str], college_quotas: Dict[str, int], final_student_matches: Dict[str, Optional[str]], student_scores: Dict[str, int]) -> Dict[str, int]:
    print("\nCollege Fill Rates and Matched Students:")
    college_fills = {college: sum(1 for match in final_student_matches.values() if match == college) for college in colleges}
    for college, fill in college_fills.items():
        print(f"{college}: {fill}/{college_quotas[college]} ({fill/college_quotas[college]*100:.2f}%)")
        matched_students = [s for s, c in final_student_matches.items() if c == college]
        print(f"  Matched students: {', '.join(matched_students[:5])}{'...' if len(matched_students) > 5 else ''}")
        print(f"  Average student score: {np.mean([student_scores[s] for s in matched_students]):.2f}")
    return college_fills

def print_additional_statistics(final_student_matches: Dict[str, Optional[str]], college_fills: Dict[str, int], college_quotas: Dict[str, int]) -> None:
    print("\nAdditional Statistics:")
    matched_students = sum(1 for match in final_student_matches.values() if match is not None)
    print(f"Total matched students: {matched_students}/{NUM_STUDENTS} ({matched_students/NUM_STUDENTS*100:.2f}%)")

    avg_college_fill_rate = sum(fill/quota for fill, quota in zip(college_fills.values(), college_quotas.values())) / NUM_COLLEGES
    print(f"Average college fill rate: {avg_college_fill_rate*100:.2f}%")

def print_average_student_satisfaction(students: List[str], final_student_matches: Dict[str, Optional[str]], student_preferences: Dict[str, List[str]]) -> None:
    avg_student_satisfaction = sum(preference_satisfaction(student, final_student_matches, student_preferences) for student in students) / NUM_STUDENTS
    print(f"Average student preference satisfaction: {avg_student_satisfaction*100:.2f}%")

def print_group_distribution(students: List[str], student_groups: Dict[str, int], final_student_matches: Dict[str, Optional[str]]) -> None:
    print("\nGroup Distribution:")
    group_counts = {group: sum(1 for s in students if student_groups[s] == group) for group in range(1, NUM_GROUPS + 1)}
    for group, count in group_counts.items():
        print(f"Group {group}: {count} students")
    
    print("\nGroup Distribution in Colleges:")
    for college in set(final_student_matches.values()):
        if college is not None:
            college_students = [s for s, c in final_student_matches.items() if c == college]
            college_group_counts = {group: sum(1 for s in college_students if student_groups[s] == group) for group in range(1, NUM_GROUPS + 1)}
            print(f"{college}:")
            for group, count in college_group_counts.items():
                print(f"  Group {group}: {count} students")

def print_results_debug_info(
    title: str,
    students: List[str],
    colleges: List[str],
    college_quotas: Dict[str, int],
    student_scores: Dict[str, int],
    final_student_matches: Dict[str, Optional[str]],
    student_preferences: Dict[str, List[str]],
    student_groups: Optional[Dict[str, int]] = None,
    college_group_quotas: Optional[Dict[str, Dict[int, int]]] = None
) -> None:
    print(f"\n{title}")
    print_college_quotas(college_quotas)
    print_lowest_scoring_students(students, student_scores, final_student_matches)
    print_highest_scoring_students(students, student_scores, final_student_matches)
    college_fills = print_college_fill_rates(colleges, college_quotas, final_student_matches, student_scores)
    print_additional_statistics(final_student_matches, college_fills, college_quotas)
    print_average_student_satisfaction(students, final_student_matches, student_preferences)
    
    if student_groups and college_group_quotas:
        print_group_distribution(students, student_groups, final_student_matches)

#####################################################################


def main() -> None:
    set_random_seed()
    
    colleges, students = generate_colleges_and_students()
    college_quotas = generate_college_quotas(colleges)
    student_scores = generate_student_scores(students)
    student_preferences = generate_all_student_preferences(students, colleges)
    college_preferences = generate_all_college_preferences(colleges, students, student_scores)

    final_student_matches = gale_shapley(student_preferences, college_preferences, college_quotas, students, colleges, process_student_application)
    print_results_debug_info("Results without groups:", students, colleges, college_quotas, student_scores, final_student_matches, student_preferences)

    student_groups_random = assign_student_groups(students)
    college_group_quotas = generate_college_group_quotas(college_quotas)
    
    final_student_matches_with_random_groups = gale_shapley(
        student_preferences, college_preferences, college_quotas, 
        students, colleges, process_student_application_with_groups,
        student_groups_random, college_group_quotas
    )
    print_results_debug_info("Results with random groups:", students, colleges, college_quotas, student_scores, final_student_matches_with_random_groups, student_preferences, student_groups_random, college_group_quotas)

    student_groups_score_based = assign_score_based_groups(students, student_scores)
    
    final_student_matches_with_score_groups = gale_shapley(
        student_preferences, college_preferences, college_quotas, 
        students, colleges, process_student_application_with_groups,
        student_groups_score_based, college_group_quotas
    )
    print_results_debug_info("Results with score-based groups:", students, colleges, college_quotas, student_scores, final_student_matches_with_score_groups, student_preferences, student_groups_score_based, college_group_quotas)
    
    generate_average_student_score_graph(colleges, final_student_matches, final_student_matches_with_random_groups, final_student_matches_with_score_groups, student_scores, "average_student_score_per_college_comparison.png")


if __name__ == "__main__":
    main()