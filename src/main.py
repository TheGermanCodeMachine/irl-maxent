import sys

def main():
    print(sys.argv[1])
    if len(sys.argv) == 1:
        import examples.original_example as example
        example.run()
    elif sys.argv[1] == 0:
        import examples.original_example as example
        example.run()
    elif sys.argv[1] == 1:
        import examples.wrong_state_example as example
        example.run()
    elif sys.argv[1] == 2:
        print("not implemented yet")
    elif sys.argv[1] == 3:
        print("not implemented yet")
    elif sys.argv[1] == 4:
        print("not implemented yet")
    elif sys.argv[1] == 5:
        print("not implemented yet")
    elif sys.argv[1] == 6:
        print("not implemented yet")
    elif sys.argv[1] == 7:
        print("not implemented yet")
    elif sys.argv[1] == 8:
        print("not implemented yet")

if __name__ == '__main__':
    main()