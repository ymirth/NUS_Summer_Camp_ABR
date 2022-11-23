var MLBitrateRule;

function MLBitrateRuleClass() {

    let factory = dashjs.FactoryMaker;
    let SwitchRequest = factory.getClassFactoryByName('SwitchRequest');
    let MetricsModel = factory.getSingletonFactoryByName('MetricsModel');
    let StreamController = factory.getSingletonFactoryByName('StreamController');
    let context = this.context;
    let instance;

    let hasgotModel = false
    let model = null
    let state = []
    
    let S_Len = 105
    S_buffer_size = []
    S_end_delay =[]
    S_rebuf=[]
    S_send_data_size=[]
    S_time_interval=[]
    S_skip_time=[]
    S_chunk_len=[]
    for (let i = 0; i < S_Len; i++) {
        S_buffer_size.push(0);
        S_end_delay.push(0);
        S_rebuf.push(0);
        S_send_data_size.push(0);
        S_time_interval.push(0);
        S_skip_time.push(0);
        S_chunk_len.push(0);
    }
    let idx = 0;
    let tb = 0.75;

    async function load_model(){
        model = await tf.loadGraphModel('https://monterosa.d2.comp.nus.edu.sg/~sws2022t4/policy_js_model/model.json');
        if(model) {hasgotModel = true};
    }

    load_model();
    function getAction(state){
        let prob = model.predict([state]);
        // prob = prob.dataSync();
        let action = 0;
        for (let i = 1; i < prob.length; i++) {
            if(prob[i]>prob[action])
                action = i;
        }
        return action;
    }

    function setup() {
    }

    const average = arr => arr.reduce((acc, val) => acc + val, 0) / arr.length;

    function getMaxIndex(rulesContext) {
        // here you can get some informations aboit metrics for example, to implement the rule
        let metricsModel = MetricsModel(context).getInstance();
        var mediaType = rulesContext.getMediaInfo().type;
        var metrics = metricsModel.getMetricsFor(mediaType, true);

        // A smarter (real) rule could need analyze playback metrics to take
        // bitrate switching decision. Printing metrics here as a reference
        console.log(metrics);
        
        state =[]
        // Get current bitrate
        let streamController = StreamController(context).getInstance();
        let abrController = rulesContext.getAbrController();
        let current = abrController.getQualityFor(mediaType, streamController.getActiveStreamInfo().id);
        

        let bufferSize = dashMetrics.getCurrentBufferLevel(mediaType,true);
        S_buffer_size[idx] = bufferSize;
        state.push(average(S_buffer_size)/10.0);//#1

        let end_delay = playbackController.getCurrentLiveLatency();
        S_end_delay[idx] = end_delay;
        state.push(average(S_end_delay)/1.8);//#2

        state.push(0.0000001);//#3
        
        let send_data_size = requests.filter(x => x.type === 'MediaSegment' && x._stream === mediaType).length;
        S_send_data_size[idx] = send_data_size;
        state.push(average(S_send_data_size)/3000.0);//#4

        let time_interval = new Date().getTime() / 1000;
        S_time_interval[idx] = time_interval;
        state.push(average(S_time_interval)/20.0);//#5

        state.push(0.0015);//#6

        let chunk_len = rulesContext.getRepresentationInfo().fragmentDuration;
        S_chunk_len[idx] = chunk_len
        state.push(average(S_chunk_len)/2000.0);//#7
        
        state.push(current/1000.0); //#8

        state.push(tb); //#9
        
        idx=(idx+1)%S_Len;
        const scheduleController = rulesContext.getScheduleController();
        const playbackController = scheduleController.getPlaybackController();

        // Ask to switch to the predict bitrate
        let switchRequest = SwitchRequest(context).create();
        ac = getAction(state);
        qua = ac % 4;
        tb = ac<=3 ? 0.5 : 1.0;
        // If already in predicted bitrate, don't do anything
        if (current === qua) {
            return SwitchRequest(context).create();
        }
        switchRequest.quality = qua;
        switchRequest.reason = 'switching to the predicted bitrate';
        switchRequest.priority = SwitchRequest.PRIORITY.STRONG;
        return switchRequest;
    }

    instance = {
        getMaxIndex: getMaxIndex
    };

    setup();

    return instance;
}

MLBitrateRuleClass.__dashjs_factory_name = 'MLBitrateRule';
MLBitrateRule = dashjs.FactoryMaker.getClassFactory(MLBitrateRuleClass);

