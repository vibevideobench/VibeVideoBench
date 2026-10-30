"""Check actual submitted media. This does not score story, typography or sound quality."""
import argparse
import json
import subprocess
import shutil
from fractions import Fraction
from pathlib import Path


def check(task, video, project, timeline):
    results = []
    def add(id, name, passed, evidence):
        results.append(dict(id=id,name=name,verdict='pass' if passed else 'fail',evidence=evidence,evaluator='program'))
    add('T01','视频、可编辑项目与时间表均存在',video.is_file() and project.is_dir() and
        any(project.iterdir()) and timeline.is_file(),{'video':str(video),'project':str(project),'timeline':str(timeline)})
    if not video.is_file():
        return results
    missing=[tool for tool in ['ffmpeg','ffprobe'] if not shutil.which(tool)]
    if missing:
        results.append({'id':'T00','name':'评测环境依赖','verdict':'unknown',
                        'evidence':{'missing':missing,'note':'测量器未运行；不可把评测环境缺失归为提交失败。'},'evaluator':'program'})
        return results
    probe = subprocess.run(['ffprobe','-v','error','-count_frames','-show_streams','-show_format','-of','json',str(video)],capture_output=True,text=True)
    if probe.returncode:
        add('T02','媒体可解析并完整解码',False,probe.stderr[-1500:]);return results
    info=json.loads(probe.stdout)
    vs=[s for s in info.get('streams',[]) if s.get('codec_type')=='video']
    if len(vs)!=1:
        add('T02','唯一视频流',False,{'video_streams':len(vs)});return results
    stream=vs[0];d=task['delivery']
    decode=subprocess.run(['ffmpeg','-v','error','-xerror','-i',str(video),'-f','null','-'],capture_output=True,text=True)
    add('T02','媒体完整解码',decode.returncode==0,decode.stderr[-1500:] or 'ffmpeg decoded to EOF without error')
    add('T03','画幅正确',(stream.get('width'),stream.get('height'))==(d['width'],d['height']),
        {'actual':[stream.get('width'),stream.get('height')],'expected':[d['width'],d['height']]})
    average=Fraction(stream.get('avg_frame_rate','0/1'));nominal=Fraction(stream.get('r_frame_rate','0/1'))
    add('T04','标称与平均帧率符合交付',average==nominal==d['fps'],{'average':str(average),'nominal':str(nominal),'expected':d['fps']})
    frames=int(stream.get('nb_read_frames',0));playtime=Fraction(frames,d['fps'])
    lo,hi=map(lambda x:Fraction(str(x)),d['duration_s'])
    add('T05','实际帧数对应的时长在题设范围',lo<=playtime<=hi,
        {'decoded_frames':frames,'duration_s_at_target_fps':float(playtime),'expected_s':d['duration_s']})
    if 'frame_count' in d:
        add('T07','实际解码总帧数符合精确要求',frames==d['frame_count'],
            {'decoded_frames':frames,'expected_frames':d['frame_count']})
    if d.get('audio_required'):
        audio=[s for s in info['streams'] if s.get('codec_type')=='audio']
        add('T06','必需声音的音频流存在',bool(audio),{'audio_streams':len(audio),'limit':'Presence is not proof of audible or correct content; model audio criteria remain required.'})
    return results


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('task_json',type=Path);p.add_argument('--video',type=Path,required=True)
    p.add_argument('--project',type=Path,required=True);p.add_argument('--timeline',type=Path,required=True)
    a=p.parse_args();t=json.loads(a.task_json.read_text())
    results=check(t,a.video,a.project,a.timeline)
    print(json.dumps({'task_id':t.get('task_id'),'checks':results,'content_and_quality':'not_evaluated'},ensure_ascii=False,indent=2))
    raise SystemExit(0 if all(r['verdict']=='pass' for r in results) else 1)
