import argparse

def main():
    parser = argparse.ArgumentParser()
    
    # 필수 positional arguments
    parser.add_argument('--config')
    parser.add_argument('--i', required=True )
    args = parser.parse_args()
    print(args)
    print(args.config)

    # parser.add_argument('input', help='입력 파일')
    # parser.add_argument('output', help='출력 파일')
    
    # # 선택적 optional arguments
    # parser.add_argument('--config', '-c', help='설정 파일')
    # parser.add_argument('--verbose', '-v', action='store_true')
    
    # args = parser.parse_args()
    
    # # 로직 실행
    # process(args.input, args.output, args.config, args.verbose)

if __name__ == '__main__':
    main()