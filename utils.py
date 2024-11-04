def print_nice_title(title: str):
    # Define the separator line
    separator = "=" * 64
    total_length = len(separator)

    # Define the padding
    padding_space = 2

    # Get the title length
    title_length = len(title)

    # Calculate num of asterisks
    num_asterisks = total_length - title_length - padding_space

    # Get the number of asterisks
    left_asterisks = num_asterisks // 2
    right_asterisks = num_asterisks - left_asterisks

    asterisk_left_side = "*" * left_asterisks
    asterisk_right_side = "*" * right_asterisks

    pretty_title = asterisk_left_side + " " + title + " " + asterisk_right_side
    print(separator)
    print(pretty_title)
    print(separator)