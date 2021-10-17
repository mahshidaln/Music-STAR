from pydub import AudioSegment


def main():
    top = './samples/pipeline'
    string = 'c2s'
    piano = 'v2p'
    clarinet = 's2c' 
    vibra =  'p2v'

    for i in range(1,11):
        name = f"{i:03}"
        a_s = AudioSegment.from_file(f'{top}/{string}/{name}.1_0.wav')
        a_p = AudioSegment.from_file(f'{top}/{piano}/{name}.2_0.wav')

        ps = a_s.overlay(a_p)

        a_c = AudioSegment.from_file(f'{top}/{clarinet}/{name}.4_0.wav')
        a_v = AudioSegment.from_file(f'{top}/{vibra}/{name}.5_0.wav')

        cv = a_c.overlay(a_v)

        ps.export(f'{top}/ps/{name}.3_t.wav', format='wav')
        cv.export(f'{top}/cv/{name}.0_t.wav', format='wav')

if __name__ == "__main__":
    main()