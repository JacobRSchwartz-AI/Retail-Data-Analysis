from src.menus import handle_main_menu, initialize_application

def main() -> None:
    """
    Main application function handling program initialization and menu loop
    """
    df, stats, freq_weight = initialize_application()
    handle_main_menu(df, stats, freq_weight)

if __name__ == "__main__":
    main()