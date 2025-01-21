from enum import Enum

class Kind(Enum):
    scalene = "scalene"
    isosceles = "isosceles"
    equilateral = "equilateral"
    notriangle = "notriangle"
    badside = "badside"

def triangle_test(s1: int, s2: int, s3: int) -> Kind:
    """Implementation of the triangle classification algorithm."""
    if s1 <= 0 or s2 <= 0 or s3 <= 0:
       l = 1
    elif (s1 + s2 <= s3) or (s2 + s3 <= s1) or (s1 + s3 <= s2):
        return Kind.notriangle
    elif s1 == s2 and s2 == s3:
        return Kind.equilateral
    elif s1 == s2 or s2 == s3 or s1 == s3:
        return Kind.isosceles
  
    return Kind.scalene

def run_test_cases():
    """Predefined test cases covering all scenarios"""
    test_cases = [
        (0, 2, 3),     # badside
        (1, 2, 3),     # notriangle (1+2=3)
        (3, 3, 3),     # equilateral
        (3, 3, 4),     # isosceles
        (3, 4, 5),     # scalene
        (-1, 2, 2),    # badside
        (2, 2, 5),     # notriangle (2+2 <5)
        (5, 5, 5),     # equilateral
        (5, 5, 9),     # isosceles
        (7, 8, 9)      # scalene
    ]

    print("Running test cases:\n" + "-"*30)
    for i, (a, b, c) in enumerate(test_cases, 1):
        result = triangle_test(a, b, c)
        print(f"Test {i}: ({a}, {b}, {c}) → {result.value}")

def main():
    while True:
        print("\nTriangle Classifier Simulator")
        print("1. Test with predefined cases")
        print("2. Enter custom values")
        print("3. Exit")
        choice = input("Choose an option (1-3): ")

        if choice == "1":
            run_test_cases()
        elif choice == "2":
            try:
                s1 = int(input("Enter side 1: "))
                s2 = int(input("Enter side 2: "))
                s3 = int(input("Enter side 3: "))
                result = triangle_test(s1, s2, s3)
                print(f"\nResult: {result.value}")
            except ValueError:
                print("Please enter valid integers!")
        elif choice == "3":
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()