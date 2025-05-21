#include <TFile.h>
#include <TTree.h>
#include <TChain.h>
#include <TMath.h>
#include <TVector3.h>
#include <TString.h>
#include <TRandom3.h>
#include <TSystem.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <map>
#include <set>

// JetCorrector
#include "/Users/lukuan/Desktop/Jet_Drift/Codes/JC/JetCorrector.h"
// Efficiency corrector
#include "/Users/lukuan/Desktop/Jet_Drift/Codes/JC/alephTrkEfficiency.h"
// Hungarian matcher (templates)
#include "Matching.h"

static alephTrkEfficiency efficiencyCorrector;

//----------------------------------------------
// constants
//----------------------------------------------
static const double kJetConeR      = 0.4;                 // cone R
static const double kMatchAngleCut = 0.2 * TMath::Pi();   // 0.2π

//======================================================================
// (pt,eta,phi) → (px,py,pz)
//======================================================================
inline void PtEtaPhi_to_PxPyPz(double pt,double eta,double phi,
                               double &px,double &py,double &pz)
{
    px = pt * std::cos(phi);
    py = pt * std::sin(phi);
    pz = pt * std::sinh(eta);
}

//======================================================================
// Compute SOP jet axis (sum-over-particles)
//======================================================================
bool ComputeSOPJetAxis(const std::vector<TVector3>& parts,
                       TVector3 &sopDir,double &sopE)
{
    TVector3 sum(0,0,0);
    for(const auto&p:parts) sum += p;
    if(sum.Mag()<1e-9) return false;
    sopDir = sum.Unit();
    sopE   = sum.Mag();
    return true;
}

//======================================================================
// Compute WTA jet axis (winner-take-all hierarchical clustering)
//======================================================================
struct Cluster{ TVector3 dir; double E; Cluster(const TVector3&d,double e):dir(d),E(e){} };

bool ComputeWTAJetAxis(const std::vector<TVector3>& parts,
                       const std::vector<float>& masses,
                       TVector3 &wtaDir,double &wtaE)
{
    size_t n=parts.size();
    if(n==0) return false;
    if(n==1){
        double p=parts[0].Mag(), m=masses[0];
        wtaE   = std::sqrt(p*p+m*m);
        wtaDir = (p<1e-12?TVector3(1,0,0):parts[0].Unit());
        return true;
    }
    std::vector<Cluster> cls; cls.reserve(n);
    for(size_t i=0;i<n;++i){
        double p=parts[i].Mag(), m=masses[i];
        cls.emplace_back((p<1e-12?TVector3(1,0,0):parts[i].Unit()),
                         std::sqrt(p*p+m*m));
    }
    while(cls.size()>1){
        double best=1e9; int ia=-1,ib=-1;
        for(size_t i=0;i<cls.size();++i)
            for(size_t j=i+1;j<cls.size();++j){
                double ang=cls[i].dir.Angle(cls[j].dir);
                if(ang<best){ best=ang; ia=(int)i; ib=(int)j; }
            }
        if(ia<0||ib<0) break;
        if(ia>ib) std::swap(ia,ib);
        double E1=cls[ia].E, E2=cls[ib].E;
        TVector3 newDir=(E1>=E2?cls[ia].dir:cls[ib].dir);
        double newE=E1+E2;
        cls.erase(cls.begin()+ib);
        cls.erase(cls.begin()+ia);
        cls.emplace_back(newDir,newE);
    }
    wtaDir=cls[0].dir; wtaE=cls[0].E; return true;
}

//======================================================================
// DoFourierAnalysis  (v1/v2/v3; with efficiency correction)
//======================================================================
bool DoFourierAnalysis(const std::vector<TVector3>& parts,
                       const std::vector<Short_t>&  charges,
                       const TVector3 &jetDir,
                       double jetPt,double jetMass,
                       int filterMode,double angleCut,int nChargedHP,
                       Int_t &outN,Float_t &outPt,Float_t &outTheta,Float_t &outE,
                       Float_t &outV1,Float_t &outV2,Float_t &outV3)
{
    double denomXY=std::sqrt(jetDir.X()*jetDir.X()+jetDir.Y()*jetDir.Y());
    if(denomXY<1e-12) return false;
    double pMag=jetPt/denomXY;
    outE   =std::sqrt(pMag*pMag+jetMass*jetMass);
    outPt  =jetPt;
    outTheta=jetDir.Theta();

    static TRandom3 rng(0);
    TVector3 ref(rng.Uniform(-1,1),rng.Uniform(-1,1),rng.Uniform(-1,1));
    if(ref.Mag()<1e-9) ref.SetXYZ(1,0,0);
    TVector3 xA=ref.Cross(jetDir); if(xA.Mag()<1e-9) xA.SetXYZ(1,0,0); xA.SetMag(1);
    TVector3 yA=jetDir.Cross(xA);  yA.SetMag(1);

    double c1=0,s1=0,c2=0,s2=0,c3=0,s3=0,sumW=0; int cnt=0;
    for(size_t i=0;i<parts.size();++i){
        if(filterMode==1 && charges[i]==0) continue;
        if(filterMode==2 && charges[i]!=0) continue;
        if(parts[i].Angle(jetDir)>angleCut) continue;

        TVector3 loc=parts[i]-jetDir*(parts[i].Dot(jetDir));
        double phi=std::atan2(loc.Dot(yA),loc.Dot(xA));
        double w=1.0;
        if(charges[i]!=0){
            double eff=efficiencyCorrector.efficiency(parts[i].Theta(),
                                                     parts[i].Phi(),
                                                     parts[i].Perp(),
                                                     (Float_t)nChargedHP);
            if(eff<1e-9) eff=1.0;
            w=1.0/eff;
        }
        c1+=w*std::cos(phi); s1+=w*std::sin(phi);
        c2+=w*std::cos(2*phi); s2+=w*std::sin(2*phi);
        c3+=w*std::cos(3*phi); s3+=w*std::sin(3*phi);
        sumW+=w; ++cnt;
    }
    if(cnt==0||sumW<1e-12) return false;
    outV1=std::sqrt(c1*c1+s1*s1)/sumW;
    outV2=std::sqrt(c2*c2+s2*s2)/sumW;
    outV3=std::sqrt(c3*c3+s3*s3)/sumW;
    outN =cnt;
    return true;
}

//======================================================================
// helper structures
//======================================================================
struct RunEventKey{ Int_t runNo,eventNo;
    bool operator<(const RunEventKey&o)const{
        return (runNo==o.runNo)?(eventNo<o.eventNo):(runNo<o.runNo);}};

struct IndexHolder{ std::vector<Long64_t> recEntries,recGenEntries,dataEntries; };

struct JetObj{ TVector3 axis; double mass; int index; };

//======================================================================
// Metric for Hungarian
//======================================================================
double MetricJetAxis(JetObj a,JetObj b){ return a.axis.Angle(b.axis); }

//======================================================================
// DoMatching_RecGen (Hungarian + kMatchAngleCut)
//======================================================================
void DoMatching_RecGen(const std::vector<JetObj>& recJets,
                       const std::vector<JetObj>& genJets,
                       Int_t runNo,Int_t eventNo,
                       Int_t &globYes,Int_t &globFail,
                       TTree *tr,
                       Int_t &b_run,Int_t &b_evt,
                       Int_t &b_recIdx,Int_t &b_genIdx,
                       Float_t &b_recPt,Float_t &b_recEta,Float_t &b_recPhi,Float_t &b_recM,
                       Float_t &b_genPt,Float_t &b_genEta,Float_t &b_genPhi,Float_t &b_genM,
                       Int_t &b_isMatched,Int_t &b_matchID)
{
    std::map<int,int> mapRG = MatchJetsHungarian(MetricJetAxis, genJets, recJets);
    std::vector<char> recoUsed(recJets.size(),0);

    // --- loop over all gen jets ------------------------------------
    for(size_t ig=0; ig<genJets.size(); ++ig){
        int ir = -1;
        auto it=mapRG.find((int)ig);
        if(it!=mapRG.end()) ir=it->second;

        b_run=runNo; b_evt=eventNo;
        if(ir>=0 && ir<(int)recJets.size()){
            recoUsed[ir]=1;

            b_recIdx=recJets[ir].index;
            b_recPt =recJets[ir].axis.Perp();
            b_recEta=recJets[ir].axis.Eta();
            b_recPhi=recJets[ir].axis.Phi();
            b_recM  =recJets[ir].mass;

            b_genIdx=genJets[ig].index;
            b_genPt =genJets[ig].axis.Perp();
            b_genEta=genJets[ig].axis.Eta();
            b_genPhi=genJets[ig].axis.Phi();
            b_genM  =genJets[ig].mass;

            double d=MetricJetAxis(genJets[ig],recJets[ir]);
            if(d<kMatchAngleCut){ b_isMatched=1; b_matchID=++globYes; }
            else                { b_isMatched=0; b_matchID=++globFail;}
        }
        else{
            // gen unmatched
            b_recIdx=-1;
            b_recPt=b_recEta=b_recPhi=b_recM=-999.f;

            b_genIdx=genJets[ig].index;
            b_genPt =genJets[ig].axis.Perp();
            b_genEta=genJets[ig].axis.Eta();
            b_genPhi=genJets[ig].axis.Phi();
            b_genM  =genJets[ig].mass;

            b_isMatched=0; b_matchID=++globFail;
        }
        tr->Fill();
    }

    // --- leftover reco jets ----------------------------------------
    for(size_t ir=0; ir<recJets.size(); ++ir){
        if(recoUsed[ir]) continue;

        b_run=runNo; b_evt=eventNo;

        b_recIdx=recJets[ir].index;
        b_recPt =recJets[ir].axis.Perp();
        b_recEta=recJets[ir].axis.Eta();
        b_recPhi=recJets[ir].axis.Phi();
        b_recM  =recJets[ir].mass;

        b_genIdx=-1; b_genPt=b_genEta=b_genPhi=b_genM=-999.f;

        b_isMatched=0; b_matchID=++globFail;
        tr->Fill();
    }
}

//======================================================================
// Main macro
//======================================================================
void AnalyzeJetAngleDist_Modified_Logical_WTA11_09_withHungarian()
{
    //------------------------------------------------------------------
    // (A) Jet energy correctors
    //------------------------------------------------------------------
    JetCorrector JEC_MC("/Users/lukuan/Desktop/Jet_Drift/Codes/JC/JEC_EEAK4_MC_20200625.txt");
    JetCorrector JEC_Data({
        "/Users/lukuan/Desktop/Jet_Drift/Codes/JC/JEC_EEAK4_MC_20200625.txt",
        "/Users/lukuan/Desktop/Jet_Drift/Codes/JC/JEC_EEAK4_DataL2_20200625.txt",
        "/Users/lukuan/Desktop/Jet_Drift/Codes/JC/JEC_EEAK4_DataL3_20200625.txt"});

    //------------------------------------------------------------------
    // (B) Build TChains  (保持原文件列表)
    //------------------------------------------------------------------
    TChain *chainRec    = new TChain("t");
    TChain *chainRecJet = new TChain("akR4ESchemeJetTree");
    for(int i=1;i<=40;++i){
        TString f=Form("/Users/lukuan/Desktop/CERNDATA/ALEPHMC/LEP1MC1994_recons_aftercut-%03d.root",i);
        chainRec->Add(f); chainRecJet->Add(f);
    }
    chainRecJet->SetTitle("jetTree"); chainRec->AddFriend(chainRecJet,"jetTree");

    TChain *chainRecGen    = new TChain("tgen");
    TChain *chainRecGenJet = new TChain("akR4ESchemeJetTree");
    for(int i=1;i<=40;++i){
        TString f=Form("/Users/lukuan/Desktop/CERNDATA/ALEPHMC/LEP1MC1994_recons_aftercut-%03d.root",i);
        chainRecGen->Add(f); chainRecGenJet->Add(f);
    }
    chainRecGenJet->SetTitle("jetTree"); chainRecGen->AddFriend(chainRecGenJet,"jetTree");

    TChain *chainData    = new TChain("t");
    TChain *chainDataJet = new TChain("akR4ESchemeJetTree");
    chainData->Add("/Users/lukuan/Desktop/CERNDATA/ALEPH/LEP1Data1994P1_recons_aftercut-MERGED.root");
    chainDataJet->Add("/Users/lukuan/Desktop/CERNDATA/ALEPH/LEP1Data1994P1_recons_aftercut-MERGED.root");
    chainData->Add("/Users/lukuan/Desktop/CERNDATA/ALEPH/LEP1Data1994P2_recons_aftercut-MERGED.root");
    chainDataJet->Add("/Users/lukuan/Desktop/CERNDATA/ALEPH/LEP1Data1994P2_recons_aftercut-MERGED.root");
    chainData->Add("/Users/lukuan/Desktop/CERNDATA/ALEPH/LEP1Data1994P3_recons_aftercut-MERGED.root");
    chainDataJet->Add("/Users/lukuan/Desktop/CERNDATA/ALEPH/LEP1Data1994P3_recons_aftercut-MERGED.root");
    chainDataJet->SetTitle("jetTree"); chainData->AddFriend(chainDataJet,"jetTree");

    //------------------------------------------------------------------
    // (C) Run / Event branches
    //------------------------------------------------------------------
    Int_t runNo_t,eventNo_t;  chainRec   ->SetBranchAddress("RunNo",&runNo_t);
    chainRec   ->SetBranchAddress("EventNo",&eventNo_t);
    Int_t runNo_tg,eventNo_tg; chainRecGen->SetBranchAddress("RunNo",&runNo_tg);
    chainRecGen->SetBranchAddress("EventNo",&eventNo_tg);
    Int_t runNo_d,eventNo_d;  chainData  ->SetBranchAddress("RunNo",&runNo_d);
    chainData  ->SetBranchAddress("EventNo",&eventNo_d);

    //------------------------------------------------------------------
    // (D) (run,event) → entry map
    //------------------------------------------------------------------
    std::map<RunEventKey,IndexHolder> runEvtMap;
    for(Long64_t i=0,n=chainRec->GetEntries(); i<n; ++i){
        chainRec->GetEntry(i);
        runEvtMap[{runNo_t,eventNo_t}].recEntries.push_back(i);
    }
    for(Long64_t i=0,n=chainRecGen->GetEntries(); i<n; ++i){
        chainRecGen->GetEntry(i);
        runEvtMap[{runNo_tg,eventNo_tg}].recGenEntries.push_back(i);
    }
    for(Long64_t i=0,n=chainData->GetEntries(); i<n; ++i){
        chainData->GetEntry(i);
        runEvtMap[{runNo_d,eventNo_d}].dataEntries.push_back(i);
    }

    //------------------------------------------------------------------
    // (E) output file & trees
    //------------------------------------------------------------------
    TFile *outFile=new TFile("JetFourierAnalysis_WTA11_09_withHungarian.root","RECREATE");

    // Matching tree
    TTree *treeMatching=new TTree("jetMatching","Rec-Gen matching");
    Int_t   m_run,m_evt,m_recIdx,m_genIdx,m_isMatched,m_matchID;
    Float_t m_recPt,m_recEta,m_recPhi,m_recM;
    Float_t m_genPt,m_genEta,m_genPhi,m_genM;
    treeMatching->Branch("RunNo",&m_run,"RunNo/I");
    treeMatching->Branch("EventNo",&m_evt,"EventNo/I");
    treeMatching->Branch("recJetIndex",&m_recIdx,"recJetIndex/I");
    treeMatching->Branch("genJetIndex",&m_genIdx,"genJetIndex/I");
    treeMatching->Branch("recJetPt",&m_recPt,"recJetPt/F");
    treeMatching->Branch("recJetEta",&m_recEta,"recJetEta/F");
    treeMatching->Branch("recJetPhi",&m_recPhi,"recJetPhi/F");
    treeMatching->Branch("recJetMass",&m_recM,"recJetMass/F");
    treeMatching->Branch("genJetPt",&m_genPt,"genJetPt/F");
    treeMatching->Branch("genJetEta",&m_genEta,"genJetEta/F");
    treeMatching->Branch("genJetPhi",&m_genPhi,"genJetPhi/F");
    treeMatching->Branch("genJetMass",&m_genM,"genJetMass/F");
    treeMatching->Branch("isMatched",&m_isMatched,"isMatched/I");
    treeMatching->Branch("matchID",&m_matchID,"matchID/I");

    // Fourier & other trees (same names as原版) ------------------------
    auto NewTree=[&](const char* n,const char* t){ return new TTree(n,t); };
    TTree *tAll_t      =NewTree("all_particle_t"           ,"All(t)");
    TTree *tCharged_t  =NewTree("charged_particle_t"       ,"Charged(t)");
    TTree *tE1_t       =NewTree("charged_particle_E1to2_t" ,"Charged 1<E<2(t)");
    TTree *tE2_t       =NewTree("charged_particle_E2to4_t" ,"Charged 2<E<4(t)");
    TTree *tE4_t       =NewTree("charged_particle_Egt4_t"  ,"Charged E>4(t)");
    TTree *tAll_tg     =NewTree("all_particle_tgen"        ,"All(tgen)");
    TTree *tCharged_tg =NewTree("charged_particle_tgen"    ,"Charged(tgen)");
    TTree *tE1_tg      =NewTree("charged_particle_E1to2_tgen","Charged 1<E<2(tgen)");
    TTree *tE2_tg      =NewTree("charged_particle_E2to4_tgen","Charged 2<E<4(tgen)");
    TTree *tE4_tg      =NewTree("charged_particle_Egt4_tgen","Charged E>4(tgen)");
    TTree *tAll_d      =NewTree("all_particle_data"        ,"All(data)");
    TTree *tCharged_d  =NewTree("charged_particle_data"    ,"Charged(data)");
    TTree *tE1_d       =NewTree("charged_particle_E1to2_data","Charged 1<E<2(data)");
    TTree *tE2_d       =NewTree("charged_particle_E2to4_data","Charged 2<E<4(data)");
    TTree *tE4_d       =NewTree("charged_particle_Egt4_data","Charged E>4(data)");

    // common branches
    Int_t runNumber,eventNumber,jetIndex,nSelParts,nGlobalParticle,nChargedHP;
    Float_t jetPt,jetTheta,jetEnergy,fourierV1,fourierV2,fourierV3,jetPhi;
    auto AddBranches=[&](TTree* tr){
        tr->Branch("runNumber",&runNumber,"runNumber/I");
        tr->Branch("eventNumber",&eventNumber,"eventNumber/I");
        tr->Branch("jetIndex",&jetIndex,"jetIndex/I");
        tr->Branch("nGlobalParticle",&nGlobalParticle,"nGlobalParticle/I");
        tr->Branch("nSelectedParts",&nSelParts,"nSelectedParts/I");
        tr->Branch("nChargedHadronsHP",&nChargedHP,"nChargedHadronsHP/I");
        tr->Branch("jetPt",&jetPt,"jetPt/F");
        tr->Branch("jetTheta",&jetTheta,"jetTheta/F");
        tr->Branch("jetEnergy",&jetEnergy,"jetEnergy/F");
        tr->Branch("fourierV1",&fourierV1,"fourierV1/F");
        tr->Branch("fourierV2",&fourierV2,"fourierV2/F");
        tr->Branch("fourierV3",&fourierV3,"fourierV3/F");
        tr->Branch("jetPhi",&jetPhi,"jetPhi/F");
    };
    for(TTree* tr : {tAll_t,tCharged_t,tE1_t,tE2_t,tE4_t,
                     tAll_tg,tCharged_tg,tE1_tg,tE2_tg,tE4_tg,
                     tAll_d,tCharged_d,tE1_d,tE2_d,tE4_d}) AddBranches(tr);

    //------------------------------------------------------------------
    // (F) particle & jet branches (保持原设置)
    //------------------------------------------------------------------
    const int MAXP=20000,MAXJ=2000;
    // rec
    Int_t nParticle_t; Float_t px_t[MAXP],py_t[MAXP],pz_t[MAXP],ms_t[MAXP]; Short_t chg_t[MAXP];
    chainRec->SetBranchAddress("nParticle",&nParticle_t);
    chainRec->SetBranchAddress("px",px_t); chainRec->SetBranchAddress("py",py_t);
    chainRec->SetBranchAddress("pz",pz_t); chainRec->SetBranchAddress("mass",ms_t);
    chainRec->SetBranchAddress("charge",chg_t);
    Int_t nref_t; Float_t jtpt_t[MAXJ],jteta_t[MAXJ],jtphi_t[MAXJ],jtm_t[MAXJ];
    chainRec->SetBranchAddress("jetTree.nref",&nref_t);
    chainRec->SetBranchAddress("jetTree.jtpt",jtpt_t);
    chainRec->SetBranchAddress("jetTree.jteta",jteta_t);
    chainRec->SetBranchAddress("jetTree.jtphi",jtphi_t);
    chainRec->SetBranchAddress("jetTree.jtm",jtm_t);
    // recGen
    Int_t nParticle_tg; Float_t px_tg[MAXP],py_tg[MAXP],pz_tg[MAXP],ms_tg[MAXP]; Short_t chg_tg[MAXP];
    chainRecGen->SetBranchAddress("nParticle",&nParticle_tg);
    chainRecGen->SetBranchAddress("px",px_tg); chainRecGen->SetBranchAddress("py",py_tg);
    chainRecGen->SetBranchAddress("pz",pz_tg); chainRecGen->SetBranchAddress("mass",ms_tg);
    chainRecGen->SetBranchAddress("charge",chg_tg);
    Int_t nref_tg; Float_t jtpt_tg[MAXJ],jteta_tg[MAXJ],jtphi_tg[MAXJ],jtm_tg[MAXJ];
    chainRecGen->SetBranchAddress("jetTree.nref",&nref_tg);
    chainRecGen->SetBranchAddress("jetTree.jtpt",jtpt_tg);
    chainRecGen->SetBranchAddress("jetTree.jteta",jteta_tg);
    chainRecGen->SetBranchAddress("jetTree.jtphi",jtphi_tg);
    chainRecGen->SetBranchAddress("jetTree.jtm",jtm_tg);
    // data
    Int_t nParticle_d; Float_t px_d[MAXP],py_d[MAXP],pz_d[MAXP],ms_d[MAXP]; Short_t chg_d[MAXP];
    chainData->SetBranchAddress("nParticle",&nParticle_d);
    chainData->SetBranchAddress("px",px_d); chainData->SetBranchAddress("py",py_d);
    chainData->SetBranchAddress("pz",pz_d); chainData->SetBranchAddress("mass",ms_d);
    chainData->SetBranchAddress("charge",chg_d);
    Int_t nref_d; Float_t jtpt_d[MAXJ],jteta_d[MAXJ],jtphi_d[MAXJ],jtm_d[MAXJ];
    chainData->SetBranchAddress("jetTree.nref",&nref_d);
    chainData->SetBranchAddress("jetTree.jtpt",jtpt_d);
    chainData->SetBranchAddress("jetTree.jteta",jteta_d);
    chainData->SetBranchAddress("jetTree.jtphi",jtphi_d);
    chainData->SetBranchAddress("jetTree.jtm",jtm_d);

    //------------------------------------------------------------------
    // (G) event loop
    //------------------------------------------------------------------
    std::set<int> runSet; for(const auto&kv:runEvtMap) runSet.insert(kv.first.runNo);
    Int_t globYes=0,globFail=0;
    int runCnt=0,totalRun=runSet.size();

    for(int currRun:runSet){
        std::set<int> evtSet;
        for(const auto&kv:runEvtMap) if(kv.first.runNo==currRun) evtSet.insert(kv.first.eventNo);
        std::cout<<"\n▶ Run "<<currRun<<" ("<<++runCnt<<"/"<<totalRun<<")\n";
        int evtCnt=0,totalEvt=evtSet.size();

        for(int evt:evtSet){
            std::cout << "   ▷ Event " << evt
          << " (" << ++evtCnt << "/" << totalEvt << ")"
          << std::endl;
            IndexHolder &ih=runEvtMap[{currRun,evt}];

            //---------------- rec jets ------------------
            std::vector<JetObj> recJets;
            for(Long64_t ent:ih.recEntries){
                chainRec->GetEntry(ent);
                runNumber=currRun; eventNumber=evt;
                if(nParticle_t<=0||nref_t<=0) continue;
                nGlobalParticle=nParticle_t;
                nChargedHP=0; for(int i=0;i<nParticle_t;++i) if(chg_t[i]!=0) ++nChargedHP;

                for(int j=0;j<nref_t;++j){
                    jetIndex=j;
                    double pt=jtpt_t[j], eta=jteta_t[j], phi=jtphi_t[j], jm=jtm_t[j];
                    if(pt<1e-12) continue;
                    double jpx,jpy,jpz; PtEtaPhi_to_PxPyPz(pt,eta,phi,jpx,jpy,jpz);
                    TVector3 seed(jpx,jpy,jpz); if(seed.Mag()<1e-12) continue; seed.SetMag(1);

                    std::vector<TVector3> parts; std::vector<Short_t> q; std::vector<float> ms;
                    parts.reserve(nParticle_t);
                    for(int ip=0;ip<nParticle_t;++ip){
                        TVector3 mom(px_t[ip],py_t[ip],pz_t[ip]);
                        if(mom.Angle(seed)<kJetConeR){ parts.push_back(mom); q.push_back(chg_t[ip]); ms.push_back(ms_t[ip]); }
                    }

                    TVector3 sop; double sopE;
                    if(parts.empty()||!ComputeSOPJetAxis(parts,sop,sopE)){ sop=seed; sopE=std::sqrt(pt*pt+jm*jm); }
                    else sop.SetMag(1);

                    TVector3 wta; double wtaE;
                    if(!ComputeWTAJetAxis(parts,ms,wta,wtaE)){ wta=seed; wtaE=std::sqrt(pt*pt+jm*jm); }
                    else wta.SetMag(1);

                    JEC_MC.SetJetE(wtaE); JEC_MC.SetJetTheta(wta.Theta()); JEC_MC.SetJetPhi(wta.Phi());
                    double Ecorr=JEC_MC.GetCorrectedE();
                    wta.SetMag(std::sqrt(Ecorr*Ecorr-jm*jm));
                    jetPhi=wta.Phi();

                    if(DoFourierAnalysis(parts,q,wta.Unit(),wta.Perp(),jm,0,kJetConeR,nChargedHP,
                                         nSelParts,jetPt,jetTheta,jetEnergy,fourierV1,fourierV2,fourierV3))
                        tAll_t->Fill();
                    if(DoFourierAnalysis(parts,q,wta.Unit(),wta.Perp(),jm,1,kJetConeR,nChargedHP,
                                         nSelParts,jetPt,jetTheta,jetEnergy,fourierV1,fourierV2,fourierV3)){
                        tCharged_t->Fill();
                        if(jetEnergy>1&&jetEnergy<2) tE1_t->Fill();
                        else if(jetEnergy>2&&jetEnergy<4) tE2_t->Fill();
                        else if(jetEnergy>4) tE4_t->Fill();
                    }
                    recJets.push_back({wta,jm,j});
                }
            }

            //---------------- gen jets ------------------
            std::vector<JetObj> genJets;
            for(Long64_t ent:ih.recGenEntries){
                chainRecGen->GetEntry(ent);
                runNumber=currRun; eventNumber=evt;
                if(nParticle_tg<=0||nref_tg<=0) continue;
                nGlobalParticle=nParticle_tg;
                nChargedHP=0; for(int i=0;i<nParticle_tg;++i) if(chg_tg[i]!=0) ++nChargedHP;

                for(int j=0;j<nref_tg;++j){
                    jetIndex=j;
                    double pt=jtpt_tg[j], eta=jteta_tg[j], phi=jtphi_tg[j], jm=jtm_tg[j];
                    if(pt<1e-12) continue;
                    double jpx,jpy,jpz; PtEtaPhi_to_PxPyPz(pt,eta,phi,jpx,jpy,jpz);
                    TVector3 seed(jpx,jpy,jpz); if(seed.Mag()<1e-12) continue; seed.SetMag(1);

                    std::vector<TVector3> parts; std::vector<Short_t> q; std::vector<float> ms;
                    for(int ip=0;ip<nParticle_tg;++ip){
                        TVector3 mom(px_tg[ip],py_tg[ip],pz_tg[ip]);
                        if(mom.Angle(seed)<kJetConeR){ parts.push_back(mom); q.push_back(chg_tg[ip]); ms.push_back(ms_tg[ip]); }
                    }

                    TVector3 wta; double wtaE;
                    if(!ComputeWTAJetAxis(parts,ms,wta,wtaE)){ wta=seed; wtaE=std::sqrt(pt*pt+jm*jm); }
                    else wta.SetMag(1);
                    wta.SetMag(std::sqrt(wtaE*wtaE-jm*jm));
                    jetPhi=wta.Phi();

                    if(DoFourierAnalysis(parts,q,wta.Unit(),wta.Perp(),jm,0,kJetConeR,nChargedHP,
                                         nSelParts,jetPt,jetTheta,jetEnergy,fourierV1,fourierV2,fourierV3))
                        tAll_tg->Fill();
                    if(DoFourierAnalysis(parts,q,wta.Unit(),wta.Perp(),jm,1,kJetConeR,nChargedHP,
                                         nSelParts,jetPt,jetTheta,jetEnergy,fourierV1,fourierV2,fourierV3)){
                        tCharged_tg->Fill();
                        if(jetEnergy>1&&jetEnergy<2) tE1_tg->Fill();
                        else if(jetEnergy>2&&jetEnergy<4) tE2_tg->Fill();
                        else if(jetEnergy>4) tE4_tg->Fill();
                    }
                    genJets.push_back({wta,jm,j});
                }
            }

            //---------------- data jets (unchanged) ---------------------
            for(Long64_t ent:ih.dataEntries){
                chainData->GetEntry(ent);
                runNumber=currRun; eventNumber=evt;
                if(nParticle_d<=0||nref_d<=0) continue;
                nGlobalParticle=nParticle_d;
                nChargedHP=0; for(int i=0;i<nParticle_d;++i) if(chg_d[i]!=0) ++nChargedHP;

                for(int j=0;j<nref_d;++j){
                    jetIndex=j;
                    double pt=jtpt_d[j], eta=jteta_d[j], phi=jtphi_d[j], jm=jtm_d[j];
                    if(pt<1e-12) continue;
                    double jpx,jpy,jpz; PtEtaPhi_to_PxPyPz(pt,eta,phi,jpx,jpy,jpz);
                    TVector3 seed(jpx,jpy,jpz); if(seed.Mag()<1e-12) continue; seed.SetMag(1);

                    std::vector<TVector3> parts; std::vector<Short_t> q; std::vector<float> ms;
                    for(int ip=0;ip<nParticle_d;++ip){
                        TVector3 mom(px_d[ip],py_d[ip],pz_d[ip]);
                        if(mom.Angle(seed)<kJetConeR){ parts.push_back(mom); q.push_back(chg_d[ip]); ms.push_back(ms_d[ip]); }
                    }

                    TVector3 wta; double wtaE;
                    if(!ComputeWTAJetAxis(parts,ms,wta,wtaE)){ wta=seed; wtaE=std::sqrt(pt*pt+jm*jm); }
                    else wta.SetMag(1);
                    JEC_Data.SetJetE(wtaE); JEC_Data.SetJetTheta(wta.Theta()); JEC_Data.SetJetPhi(wta.Phi());
                    double Ecorr=JEC_Data.GetCorrectedE();
                    wta.SetMag(std::sqrt(Ecorr*Ecorr-jm*jm));
                    jetPhi=wta.Phi();

                    if(DoFourierAnalysis(parts,q,wta.Unit(),wta.Perp(),jm,0,kJetConeR,nChargedHP,
                                         nSelParts,jetPt,jetTheta,jetEnergy,fourierV1,fourierV2,fourierV3))
                        tAll_d->Fill();
                    if(DoFourierAnalysis(parts,q,wta.Unit(),wta.Perp(),jm,1,kJetConeR,nChargedHP,
                                         nSelParts,jetPt,jetTheta,jetEnergy,fourierV1,fourierV2,fourierV3)){
                        tCharged_d->Fill();
                        if(jetEnergy>1&&jetEnergy<2) tE1_d->Fill();
                        else if(jetEnergy>2&&jetEnergy<4) tE2_d->Fill();
                        else if(jetEnergy>4) tE4_d->Fill();
                    }
                }
            }

            //---------------- matching -------------------
            if(!recJets.empty() && !genJets.empty()){
                DoMatching_RecGen(recJets,genJets,currRun,evt,
                                  globYes,globFail,
                                  treeMatching,
                                  m_run,m_evt,m_recIdx,m_genIdx,
                                  m_recPt,m_recEta,m_recPhi,m_recM,
                                  m_genPt,m_genEta,m_genPhi,m_genM,
                                  m_isMatched,m_matchID);
            }
        }
    }

    //------------------------------------------------------------------
    // (H) write & clean
    //------------------------------------------------------------------
    outFile->cd();
    for(TTree* tr:{tAll_t,tCharged_t,tE1_t,tE2_t,tE4_t,
                   tAll_tg,tCharged_tg,tE1_tg,tE2_tg,tE4_tg,
                   tAll_d,tCharged_d,tE1_d,tE2_d,tE4_d}) tr->Write();
    treeMatching->Write();
    outFile->Close();

    delete chainRec; delete chainRecJet;
    delete chainRecGen; delete chainRecGenJet;
    delete chainData; delete chainDataJet;

    std::cout<<"\nAll done. Output => JetFourierAnalysis_WTA11_09_withHungarian.root\n";
}

